package com.pjottersstuff.ocrdemo;

import android.os.AsyncTask;

public class OCRProcessor extends AsyncTask<Void, Void, Void> {
	static {
		System.loadLibrary("lept");
		System.loadLibrary("tess");

		System.loadLibrary("ocr-demo");
	}

	private final String _filename;

	public OCRProcessor(String filename) {
		_filename = filename;
	}

	private native void processImage(String filename);

	@Override
	protected Void doInBackground(Void... params) {
		processImage(_filename);

		return null;
	}

	
}
