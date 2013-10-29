#include <jni.h>
#include <android/log.h>
#include <vector>
#include <cmath>
#include <list>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// largest side in image that is actually processed
#define MAX_IMAGE_EDGE 640

// canny filter settings
#define CANNY_LOW_THRESHOLD 70
#define CANNY_HIGH_THRESHOLD 120
#define CANNY_KERNEL_SIZE 3

// swt settings
#define SWT_TRACE_STEP_SCALE 0.2f

// comment this out if you're expecting to find light text on a dark background; if enabled, only dark text on a light background will be detected
#define SWT_DARK_ON_LIGHT 1

// component extraction settings
#define RATIO 3.0f

// helper structs for stroke width transform

#include <algorithm>
#include <iterator>
#include <map>
#include <utility>
#include <set>

#include "../opencv/native/jni/include/opencv2/core/core.hpp"
#include "../opencv/native/jni/include/opencv2/core/operations.hpp"
#include "../opencv/native/jni/include/opencv2/core/types_c.h"
#include "../opencv/native/jni/include/opencv2/highgui/highgui_c.h"
#include "../opencv/native/jni/include/opencv2/imgproc/types_c.h"

struct swtPoint {
	int x, y;

	float strokeWidth;

	swtPoint() :
			swtPoint(0, 0) {
	}
	swtPoint(int x, int y) :
			x(x), y(y), strokeWidth(0.0f) {
	}
};

struct swtRay {
	swtPoint start, end;

	std::vector<swtPoint> points;

	swtRay(int x, int y) :
			swtRay(swtPoint(x, y)) {
	}

	swtRay(const swtPoint &point) {
		start = point;
		points.reserve(64);
		points.push_back(start);
	}

	void add(cv::Vec2i v) {
		points.push_back(swtPoint(v[0], v[1]));
	}

	float length() const {
		return std::sqrt(
				(start.x - end.x) * (start.x - end.x)
						+ (start.y - end.y) * (start.y - end.y));
	}
};

// helper structs for component extraction

struct component {
	std::vector<cv::Point2i> points;

	float meanWidth;

	component() {
		dirty = false;
	}

	void add(int x, int y) {
		points.push_back(cv::Point2i(x, y));
		dirty = true;
	}

	const cv::Rect &getBoundingBox() const {
		if (dirty)
			update();
		return bbox;
	}

	const cv::Point2i &getCenter() const {
		if (dirty)
			update();
		return center;
	}

	float getSize() const {
		if (dirty)
			update();
		return size;
	}

private:
	cv::Point2i center;
	cv::Rect bbox;
	float size;
	bool dirty;
	void update() const {
		component *casted = const_cast<component*>(this);
		casted->dirty = false;
		casted->bbox = cv::boundingRect(points);
		casted->center = cv::Point2i(bbox.x + bbox.size().width / 2,
				bbox.y + bbox.size().height / 2);
		casted->size = std::sqrt(
				bbox.width * bbox.width + bbox.height * bbox.height);
	}
};

// helper struct for chain extraction
struct potentialChain {

	std::set<unsigned int> components;
	float angle;
};

typedef std::vector<component> chain;

bool sortSwt(const swtPoint &p1, const swtPoint &p2) {
	return p1.strokeWidth < p2.strokeWidth;
}

cv::Mat resize(cv::Mat mat) {

	// resize image so that its largest side is no more than MAX_IMAGE_EDGE
	float scale;
	if (mat.size().width > mat.size().height)
		scale = float(MAX_IMAGE_EDGE) / mat.size().width;
	else
		scale = float(MAX_IMAGE_EDGE) / mat.size().height;

	cv::resize(mat, mat, cv::Size(), scale, scale, CV_INTER_AREA);

	return mat;
}

cv::Mat decolorize(cv::Mat mat) {

	// convert to grayscale and enhance contrast
	cv::Mat gray(mat.size(), CV_8UC1);
	cv::cvtColor(mat, gray, CV_RGB2GRAY, 1);
	cv::equalizeHist(gray, gray);

	return gray;
}

template<typename T> T at(const cv::Mat &img, const cv::Vec2i &coords) {
	return img.at<T>(coords[1], coords[0]);
}

cv::Mat swt(cv::Mat mat) {

	// convert the image to floating point and blur it
	cv::Mat gaussianImage(mat.size(), CV_32FC1);
	mat.convertTo(gaussianImage, CV_32FC1, 1.0f / 255.0f);
	cv::GaussianBlur(mat, mat, cv::Size(3, 3), 0);

	// calculate its gradients
	cv::Mat gradX(mat.size(), CV_32FC1);
	cv::Mat gradY(mat.size(), CV_32FC1);
	cv::Sobel(gaussianImage, gradX, CV_32FC1, 1, 0);
	cv::Sobel(gaussianImage, gradY, CV_32FC1, 0, 1);

	// make some space in memory
	gaussianImage.release();

	// find the edges
	cv::Canny(mat, mat, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD,
	CANNY_KERNEL_SIZE, true);

	// swt fun time! \O/
	std::vector<swtRay> rays;
	rays.reserve(8096);

	cv::Mat swtImage(mat.size(), CV_32FC1, -1);

	// loop through all pixels in image
	for (int y = 0; y < mat.size().height; ++y) {
		const unsigned char *edgePtr = mat.ptr(y);
		for (int x = 0; x < mat.size().width; ++x) {

			if (*edgePtr) {

				// there is an edge at the current pixel! start raytracing!

				// grab the gradient at this pixel. Note that the order of the parameters in 'at' is no mistake, for some confusing reason its order is reversed from what is expected
				cv::Vec2f gradient;
				gradient[0] = gradX.at<float>(y, x);
				gradient[1] = gradY.at<float>(y, x);

				__android_log_print(ANDROID_LOG_DEBUG, "OCR",
						"Start trace at %dx%d  (%fx%f)", x, y, gradient[0],
						gradient[1]);

				if ((gradient[0] * gradient[0] + gradient[1] * gradient[1])
						< 0.01f) {
					++edgePtr;
					continue;
				}

				// tracing coordinates are kept in floating point, since the gradient at this spot can be assumed to not be orthogonal
				cv::Vec2f traceCoords(float(x) + 0.5f, float(y) + 0.5f);

				// however, the actual values read from the edge map only change if the integer value of the tracing coordinates change, so store those as well
				cv::Vec2i prevTraceCoords(traceCoords);

				// normalize the gradient
				gradient = cv::normalize(gradient);

#ifdef SWT_DARK_ON_LIGHT
				gradient = -gradient;
#endif

				// yay, create a ray to trace! \O/
				swtRay ray(x, y);

				while (true) {
					traceCoords += gradient * SWT_TRACE_STEP_SCALE;

					// if the integer coordinates of our trace hasn't changed, we don't really need to do anything
					cv::Vec2i intTraceCoords(traceCoords);
					if (intTraceCoords != prevTraceCoords) {
						prevTraceCoords = intTraceCoords;

						// ray has left the image; for simplicity's sake, the entire ray is dropped instead of clipped
						if (intTraceCoords[0] < 0 || intTraceCoords[1] < 0
								|| intTraceCoords[0] >= mat.size().width
								|| intTraceCoords[1] >= mat.size().height) {

							__android_log_print(ANDROID_LOG_DEBUG, "OCR",
									"Finish trace at %dx%d due to leaving image",
									intTraceCoords[0], intTraceCoords[1]);
							break;
						}

						// add new location to ray
						ray.add(intTraceCoords);

						// discovered another edge?
						if (at<uchar>(mat, intTraceCoords)) {

							// the gradient at the current location needs to go into roughly the opposite direction of the one the ray is following (or at least perpendicular)
							// first, find the current gradient
							cv::Vec2f curGradient;
							curGradient[0] = at<float>(gradX, intTraceCoords);
							curGradient[1] = at<float>(gradY, intTraceCoords);

							curGradient = cv::normalize(curGradient);

#ifdef SWT_DARK_ON_LIGHT
							curGradient = -curGradient;
#endif

							__android_log_print(ANDROID_LOG_DEBUG, "OCR",
									"End trace at %dx%d", intTraceCoords[0],
									intTraceCoords[1]);

							// secondly, find the angle between the gradients, and verify that it's larger than 90 degrees
							if (gradient.dot(-curGradient) >= 0) {
								ray.add(intTraceCoords);
								ray.end = swtPoint(intTraceCoords[0],
										intTraceCoords[1]);

								float length = ray.length();

								// write out the stroke width at all points in the ray
								for (swtPoint &point : ray.points) {
									float cur = swtImage.at<float>(point.y,
											point.x);
									if (cur < 0)
										swtImage.at<float>(point.y, point.x) =
												length;
									else
										swtImage.at<float>(point.y, point.x) =
												std::min(cur, length);
								}

								rays.push_back(ray);
								__android_log_print(ANDROID_LOG_DEBUG, "OCR",
										"Added ray with length %f", length);

							} else {
								__android_log_print(ANDROID_LOG_DEBUG, "OCR",
										"Dropped trace due to angle (%fx%f) vs (%fx%f) = %f",
										gradient[0], gradient[1],
										curGradient[0], curGradient[1],
										gradient.dot(-curGradient));
							}

							break;
						}

					}

				}

			}

			++edgePtr;

		}
	}

	// find out if any rays are overlapping, and make sure the widths in the image are set to their lowest median value
	for (swtRay & ray : rays) {
		for (swtPoint &point : ray.points)
			point.strokeWidth = swtImage.at<float>(point.y, point.x);

		std::sort(ray.points.begin(), ray.points.end(), sortSwt);

		float median = ray.points[ray.points.size() / 2].strokeWidth;
		for (swtPoint &point : ray.points) {
			swtImage.at<float>(point.y, point.x) = std::min(median,
					swtImage.at<float>(point.y, point.x));
		}

	}

	// yay, we have the swt'ed image!

	return swtImage;
}

#define CORRECT_RATIO(current, comparedTo) (comparedTo > 0) && (((current) / (comparedTo)) <= RATIO) || (((comparedTo) / (current) <= RATIO))

#define COMPARE_PIXEL(ptr, offset) { \
	float pix = *(ptr + offset); \
	if (CORRECT_RATIO(*ptr, pix)) { edges[i].push_back(i + offset); edges[i + offset].push_back(i);} \
}

template<typename T> using sparse_array = std::map<unsigned int, T>;

std::vector<component> extractComponents(cv::Mat swtImage) {

	sparse_array<std::vector<int>> edges;

	// for each pixel in the swt image, find out if its neighbours have a strokewidth differing with a factor of at most 3
	int i = 0;
	const int width = swtImage.size().width;
	const int height = swtImage.size().height;

	for (int y = 0; y < height; ++y) {
		const float *ptrIn = (const float*) swtImage.ptr(y);
		for (int x = 0; x < width; ++x, ++i, ++ptrIn) {
			if (*ptrIn >= 0) {
				if (x < width - 1) {
					// right pixel
					COMPARE_PIXEL(ptrIn, 1);
					// bottom right pixel
					if (y < height - 1)
						COMPARE_PIXEL(ptrIn, 1 + width);
				}

				if (y < height - 1) {
					// bottom pixel
					COMPARE_PIXEL(ptrIn, width);
					// bottom left pixel
					if (y > 0)
						COMPARE_PIXEL(ptrIn, width - 1);
				}

			}

		}

	}

	__android_log_print(ANDROID_LOG_DEBUG, "OCR", "Found %d edges",
			edges.size());

	// find connected edges using depth first search
	sparse_array<int> groups;
	int groupCount = 0;
	std::vector<unsigned int> stack;

	for (auto pair : edges) {
		// no group has been assigned to this vertex yet
		if (!groups.count(pair.first)) {

			stack.push_back(pair.first);
			do {
				unsigned int evaluating = stack.back();
				stack.pop_back();
				groups[evaluating] = groupCount;

				for (unsigned int vertex : edges[evaluating])
					if (!groups.count(vertex)) // guards against cycles
						stack.push_back(vertex);

			} while (!stack.empty());

			groupCount++;

		}

	}

	__android_log_print(ANDROID_LOG_DEBUG, "OCR", "Found %d components",
			groupCount);

	// now extract the components
	std::vector<component> components(groupCount, component());
	for (auto pair : groups) {
		components[pair.second].add(pair.first % width, pair.first / width);
	}

	return components;

}

bool filterComponent(cv::Mat img, component & c) {
	// filters the components based their geometric properties; returns true if the component should be filtered

	// first, verify its size; ditch components that are too big or too small
	int imgSize = img.size().width * img.size().height;
	int componentSize = c.getBoundingBox().width * c.getBoundingBox().height;

	float ratio = (float) componentSize / (float) imgSize;
	if (ratio < 0.0005 || ratio > 0.02) {
		__android_log_print(ANDROID_LOG_DEBUG, "OCR", "Size ditch %f (too %s)",
				ratio, (ratio > 0.02 ? "large" : "small"));

		return true;
	}

	// check component's aspect ratio
	float aspect = std::min(
			(float) c.getBoundingBox().width
					/ (float) c.getBoundingBox().height,
			(float) c.getBoundingBox().height
					/ (float) c.getBoundingBox().width);
	if (aspect < 0.1) {
		__android_log_print(ANDROID_LOG_DEBUG, "OCR", "Aspect ditch %f",
				aspect);
		return true;
	}

	// calculate the component's width variation and occupation rate
	float totalWidth = 0;
	int occupyCount = 0;
	for (int y = c.getBoundingBox().y;
			y < c.getBoundingBox().y + c.getBoundingBox().height; ++y)
		for (int x = c.getBoundingBox().x;
				x < c.getBoundingBox().x + c.getBoundingBox().width; ++x) {
			float width = img.at<float>(y, x);
			if (width > 0) {
				occupyCount++;
				totalWidth += width;
			}
		}

	// we can already check for occupation ratio, so do so
	float occupationRatio = (float) occupyCount / (float) componentSize;
	if (occupationRatio < 0.1) {
		__android_log_print(ANDROID_LOG_DEBUG, "OCR", "Occupation ditch %f",
				occupationRatio);
		return true;
	}

	// now calculate the width devation
	float meanWidth = totalWidth / occupyCount;
	float sum = 0.0f;
	for (int y = c.getBoundingBox().y;
			y < c.getBoundingBox().y + c.getBoundingBox().height; ++y)
		for (int x = c.getBoundingBox().x;
				x < c.getBoundingBox().x + c.getBoundingBox().width; ++x) {
			float width = img.at<float>(y, x);
			if (width > 0) {
				float diff = width - meanWidth;
				sum += diff * diff;
			}
		}

	float deviation = std::sqrt(sum / (float) occupyCount);
	float widthVariation = deviation / meanWidth;
	if (widthVariation < 0 || widthVariation > 1) {
		__android_log_print(ANDROID_LOG_DEBUG, "OCR",
				"Width variation ditch %f", widthVariation);
		return true;
	}

	// this is used later during chain generation, so store it here
	c.meanWidth = meanWidth;

	return false;
}

void filterComponents(cv::Mat img, std::vector<component> &components) {
	for (auto it = components.begin(); it != components.end();)
		if (filterComponent(img, *it))
			it = components.erase(it);
		else
			++it;
}

bool isPotentialPair(const component &c1, const component &c2) {

	// the mean stroke width of both components should be similar
	float swtRatio = std::max(c1.meanWidth / c2.meanWidth,
			c2.meanWidth / c1.meanWidth);
	if (swtRatio > 2.0)
		return false;

	// the size of the components should be similar
	float s1 = c1.getSize();
	float s2 = c2.getSize();
	float sizeRatio = std::max(s1 / s2, s2 / s1);
	if (sizeRatio > 2.5)
		return false;

	// the components should be near each other
	cv::Point2i center1 = c1.getCenter();
	cv::Point2i center2 = c2.getCenter();
	cv::Point2i relative = center2 - center1;
	float distance = std::sqrt(
			relative.x * relative.x + relative.y * relative.y);
	if (distance > 0.8 * (s1 + s2))
		return false;


	__android_log_print(ANDROID_LOG_DEBUG, "OCR", "Paired! swt: %f size: %f distance: %f size1: %f size2: %f", swtRatio, sizeRatio, distance, s1,s2);


	return true;
}

float calculateSimilarity(const potentialChain& chain1,
		const potentialChain &chain2) {
	bool found = false;
	for (unsigned int c1 : chain1.components)
		for (unsigned int c2 : chain2.components)
			if (c1 == c2) {
				found = true;
				break;
			}
	if (!found)
		return 0;

	float angleDiff = std::abs(chain1.angle - chain2.angle);

	if (angleDiff >= M_PI / 16)
		return 0;
	float So = 1 - (angleDiff / (M_PI / 2));

	int c1 = std::abs(chain1.components.size() - chain2.components.size());
	int c2 = chain1.components.size() + chain2.components.size();
	float Sp = (float) c1 / c2;

	float sigma = 0.5 * So + 0.5 * Sp;

	return sigma;
}

std::vector<chain> chainComponents(const std::vector<component> &components) {

	std::vector<potentialChain> chains;

	__android_log_print(ANDROID_LOG_DEBUG, "OCR", "Start pairing %d components",
			components.size());

	// tries to build chains of components that are supposedly part of the same word
	// first, pair the components if they are potentially part of the same word
	for (int i = 0; i < components.size(); ++i)
		for (int j = i + 1; j < components.size(); ++j)
			if (isPotentialPair(components[i], components[j])) {
				potentialChain chain;
				chain.components.insert(i);
				chain.components.insert(j);

				// calculate angle of vector between components and (0, -1)
				double dx = components[j].getCenter().x
						- components[i].getCenter().x;
				double dy = components[j].getCenter().y
						- components[i].getCenter().y;
				chain.angle = -dy / sqrt((dx * dx + dy * dy) + 1e-10);

				chains.push_back(chain);
			}

	__android_log_print(ANDROID_LOG_DEBUG, "OCR", "Paired %d chains",
			chains.size());

	// then, try and find the most sensible pairs
	std::vector<bool> dropped(chains.size(), false);
	float maxSimilarity;
	do {
		maxSimilarity = 0;
		std::pair<unsigned int, unsigned int> bestPair;

		for (int i = 0; i < chains.size(); ++i)
			if (!dropped[i])
				for (int j = i + 1; j < chains.size(); ++j)
					if (!dropped[j]) {
						float similarity = calculateSimilarity(chains[i],
								chains[j]);
						if (similarity > maxSimilarity) {
							maxSimilarity = similarity;
							bestPair = std::pair<unsigned int, unsigned int>(i,
									j);
						}
					}

		if (maxSimilarity > 0) {
			potentialChain &chain1 = chains[bestPair.first];
			potentialChain &chain2 = chains[bestPair.second];
			dropped[bestPair.first] = true;
			chain2.components.insert(chain1.components.begin(),
					chain1.components.end());
		}

	} while (!chains.empty() && maxSimilarity > 0);

	// yay, potential chains! make them into actual ones! ;)
	std::vector<chain> result;
	for (int i = 0; i < chains.size(); ++i)
		if (!dropped[i]) {
			chain ch;
			for (unsigned int comp : chains[i].components)
				ch.push_back(components[comp]);

			result.push_back(ch);
		}

	__android_log_print(ANDROID_LOG_DEBUG, "OCR", "Resulted in %d chains",
			result.size());

	return result;

}

////////// for debugging output; not part of the algorithm

cv::Mat normalize(cv::Mat swtImage) {
	cv::Mat out(swtImage.size(), CV_8UC1);

	float min = MAXFLOAT;
	float max = 0.0f;
	for (int y = 0; y < swtImage.size().height; ++y) {
		const float *ptr = (const float*) swtImage.ptr(y);
		for (int x = 0; x < swtImage.size().width; ++x) {
			if (*ptr > 0) {
				min = std::min(min, *ptr);
				max = std::max(max, *ptr);
			}
			++ptr;
		}
	}

	float diff = max - min;
	for (int y = 0; y < swtImage.size().height; ++y) {
		const float *ptrIn = (const float*) swtImage.ptr(y);
		unsigned char *ptrOut = out.ptr(y);
		for (int x = 0; x < swtImage.size().width; ++x) {

			if (*ptrIn < 0)
				*ptrOut = 255;
			else
				*ptrOut = ((*ptrIn - min) / diff) * 255.0f;

			++ptrIn;
			++ptrOut;
		}

	}

	cv::cvtColor(out, out, CV_GRAY2RGB);

	return out;
}

cv::Mat drawComponents(cv::Mat img, const std::vector<component> &components) {
	for (const component & c : components) {
		cv::rectangle(img, c.getBoundingBox(), cv::Scalar(0, 0, 255));
	}

	return img;
}

cv::Mat drawChains(cv::Mat mat, const std::vector<chain> &chains) {
	for (const chain & ch : chains) {
		for (int i = 1; i < ch.size(); ++i) {
			cv::line(mat, ch[i - 1].getCenter(), ch[i].getCenter(),
					cv::Scalar(0, 255, 0));
		}
	}
	return mat;
}

#ifdef __cplusplus
extern "C" {
#endif

void Java_com_pjottersstuff_ocrdemo_OCRProcessor_processImage(JNIEnv *env,
		jobject object, jstring filename) {

	const char * str = env->GetStringUTFChars(filename, nullptr);
	cv::Mat mat = cv::imread(str, CV_LOAD_IMAGE_COLOR);

	if (mat.size().width > MAX_IMAGE_EDGE || mat.size().width > MAX_IMAGE_EDGE)
		mat = resize(mat);

	mat = decolorize(mat);

	mat = swt(mat);

	std::vector<component> components = extractComponents(mat);

	filterComponents(mat, components);

	std::vector<chain> chains = chainComponents(components);

	mat = normalize(mat);

	mat = drawComponents(mat, components);

	mat = drawChains(mat, chains);

	cv::imwrite(str, mat);

	__android_log_print(ANDROID_LOG_DEBUG, "OCR-DEMO", "Logtest! %s %dx%d", str,
			mat.size().width, mat.size().height);

	env->ReleaseStringUTFChars(filename, str);
}

#ifdef __cplusplus
}
#endif

