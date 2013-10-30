package com.pjottersstuff.ocrdemo;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import android.content.Context;
import android.os.AsyncTask;
import android.util.Log;

public class OCRProcessor extends AsyncTask<Void, Void, List<String>> {
	static {
		System.loadLibrary("lept");
		System.loadLibrary("tess");
		System.loadLibrary("ocr-demo");
	}

	private final String _filename;
	private final Context _context;

	public OCRProcessor(Context context, String filename) {
		_filename = filename;
		_context = context;
	}

	private native String[] processImage(String filename, String dataPath);

	@Override
	protected List<String> doInBackground(Void... params) {

		String dataPath = expandDataIfNeeded(_context);

		String[] result = processImage(_filename, dataPath);

		List<String> list = new ArrayList<String>();
		if (result != null)
			for (String str : result)
				list.add(str);

		return list;
	}

	private String expandDataIfNeeded(Context context) {

		File path = context.getFilesDir();
		File tessData = new File(path, "tessdata");
		if (!tessData.exists())
			tessData.mkdir();
		
		File trained = new File(tessData, "eng.traineddata");
		if (!trained.exists()) {

			try {
				InputStream is = context.getResources().getAssets()
						.open("tesseract.zip");

				ZipInputStream zin = new ZipInputStream(is);
				ZipEntry ze = null;
				byte[] buffer = new byte[4096];
				while ((ze = zin.getNextEntry()) != null) {

					File out = new File(tessData, ze.getName());
					FileOutputStream fout = new FileOutputStream(out);
					for (int c = zin.read(buffer); c > 0; c = zin.read(buffer))
						fout.write(buffer, 0, c);

					zin.closeEntry();
					fout.close();

				}
				zin.close();
			} catch (Exception e) {
				Log.e("OCR TEST", "Error while expanding tesseract data", e);
			}
		}
		
		return path.getAbsolutePath();
	}

}
