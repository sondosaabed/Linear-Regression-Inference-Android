package com.google.tflite.basicmodel;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.widget.EditText;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private Interpreter tflite;
    private final Interpreter.Options tfliteoptions = new Interpreter.Options();
    private TextView resultTextView;
    private EditText editText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        resultTextView = findViewById(R.id.result);
        editText = findViewById(R.id.editText);

        try {
            MappedByteBuffer tflitemodel = loadModelfile(this);
            tfliteoptions.setNumThreads(1);
            assert tflitemodel != null;
            tflite = new Interpreter(tflitemodel, tfliteoptions);
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        editText.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
            }

            @Override
            public void afterTextChanged(Editable s) {
                doInference();
            }
        });
    }

    @SuppressLint("SetTextI18n")
    private void doInference() {
        if (editText.getText().toString().isEmpty()) {
            resultTextView.setText(R.string.result_0);
            return;
        }

        float intValue = Float.parseFloat(editText.getText().toString());
        float[] inputVal = new float[]{intValue};
        ByteBuffer outputVal = ByteBuffer.allocateDirect(4);
        outputVal.order(ByteOrder.nativeOrder());
        tflite.run(inputVal, outputVal);
        outputVal.rewind();
        float prediction = outputVal.getFloat();
        resultTextView.setText("Result: " + prediction);
    }

    private MappedByteBuffer loadModelfile(Activity activity) {
        try {
            AssetFileDescriptor fileDescriptor = activity.getAssets().openFd("model.tflite");
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
