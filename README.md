ocr-demo
========

Demonstration code for the OCR-presentation I gave at Droidcon UK 2013.

This project is a proof of concept/demo code of an OCR-implementation in Android. It is the companion piece of a presentation I (Pjotter Tommassen) gave at Droidcon UK 2013, the visual support of which can be found here: http://prezi.com/jdwz8spthvom/implementing-ocr-in-android-applications/

For those not interested in the presentation, it is basically an implementation of the following articles:
* B. Epshtein, E. Ofek, and Y. Wexler. Detecting text in natural scenes with stroke width transform. In Proc. CVPR, 2010.
* Cong Yao, Xiang Bai, Wenyu Liu, Yi Ma, Zhuowen Tu. Detecting Texts of Arbitrary Orientations in Natural Images. In Proc. CVPR, 2012.
(though the latter one is only implemented partially; characteristic scales aren't used, and the rest of the algorithm is updated to take this into account)

This is most definitely not the implementation mentioned during the presentation (since that one was built for a customer and therefore NDA'ed and copyrighted and such), but a reimplementation I did during my spare time. To prevent any resemblance to the original code, I skipped quite a lot of optimizations as well as basic code structuring, so this is far from production ready, and not a representation of my coding capabilities.. :'). However, it should be legible and usable as a starting point for your own implementations.

The usage of the application is quite boring; start the application, press the 'Take Picture' button, and snap a picture. OCR processing should start, and a picture of the analyzed result will be returned, as well as the recognized text blocks in Toast messages.

The accuracy of the OCR algorithm isn't entirely up to snuff. I'm not sure I can improve it too much without breaking a NDA or something, but here are a few pointers to get you started, in case you want to use this code:
* chainComponents sometimes tries to link two lines of text, while it should only consider characters in the same line.
* Tesseract tries really hard to recognize ANYTHING, meaning that random noise will be recognized as punctuation marks. Since this is rather silly, I fed a whitelist containing only alphanumerics to tesseract. However, it is possible to remove most of the noise.
* The components are abused as axis aligned bounding boxes; rotated best-fit boxes will drastically improve parts of the algorithm.
* The images extracted using the chains aren't rotated to better match the orientation of the chain.

Also, the variables of the algorithm have now been tweaked to recognize the text 'PONIES ROCK!' from my further nearly blank computer screen under artificial light conditions; it will probably not recognize text in completely different conditions (e.g. pages of a book being read in bright sunlight). With some tweaking, however, it should be able to do so! :)  

There are some bugs that I'm not sure I'm allowed to fix (since I need to avoid resemblance with the original implementation), but can and probably will impact the usage of this application. Fixing it should be quite easy though:
* The orientation of the extracted text isn't adjusted before being put through Tesseract. Also, the photo being taken isn't oriented according to how the device was held while the picture was being taken. This results in the somewhat user-unfriendly situation that the device should be held in landscape mode with the bottom pointing to the right when taking a picture, and the text should be oriented to show up right-side-up in the picture. The processed image that appears after the picture is shown should have the text displayed correctly when the device is held in NON-upside-down portrait mode.
* There is a weird bug when the camera is taking a picture, and some other application interrupts it. This may cause the CameraActivity to be closed, thus wiping the _image variable when its being reconstructed, which results in bad stuff in onActivityResult.. :')

You can basically use this code for whatever you want, with some limitations; see the LICENSE-file for details.

Anyways, I hope this is of some help to someone! :) 

   