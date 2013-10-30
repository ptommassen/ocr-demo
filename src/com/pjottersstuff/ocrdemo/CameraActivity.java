package com.pjottersstuff.ocrdemo;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.ImageView;
import android.widget.Toast;

import com.pjottersstuff.ocr_demo.R;

public class CameraActivity extends Activity {

	private static final int TAKE_IMAGE_REQUEST_CODE = 1337;

	private ImageView _imageView;
	private File _image;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		setContentView(R.layout.activity_camera);

		findViewById(R.id.picture_button).setOnClickListener(
				new OnClickListener() {

					@Override
					public void onClick(View v) {
						dispatchTakePictureIntent();
					}
				});

		_imageView = (ImageView) findViewById(R.id.image);
	}

	private void dispatchTakePictureIntent() {
		Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
		_image = createImageFile();
		takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT,
				Uri.fromFile(_image));
		startActivityForResult(takePictureIntent, TAKE_IMAGE_REQUEST_CODE);
	}

	@Override
	protected void onActivityResult(int requestCode, int resultCode,
			Intent intent) {
		if (requestCode == TAKE_IMAGE_REQUEST_CODE && resultCode == RESULT_OK
				&& _image != null) {

			Toast.makeText(getBaseContext(), "Started processing",
					Toast.LENGTH_LONG).show();

			AsyncTask<Void, Void, List<String>> task = new OCRProcessor(this,
					_image.getAbsolutePath()) {
				@Override
				protected void onPostExecute(List<String> result) {
					_imageView.setImageURI(Uri.fromFile(_image));

					for (String str : result)
						Toast.makeText(getBaseContext(), "OCRed: " + str,
								Toast.LENGTH_LONG).show();
				}
			};
			task.execute();

		}
	}

	private File createImageFile() {
		try {
			File storageDir = new File(
					Environment
							.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
					"ocr-demo");
			if (!storageDir.exists())
				storageDir.mkdirs();
			String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss")
					.format(new Date());
			String imageFileName = "ocr" + timeStamp + "_";
			File image = File.createTempFile(imageFileName, ".jpg", storageDir

			);

			return image;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

}
