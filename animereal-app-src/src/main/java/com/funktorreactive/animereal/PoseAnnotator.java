package com.funktorreactive.animereal;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.text.SpannableStringBuilder;
import android.util.Log;

//import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Class for annotating poses using TFLite
 * Abstract class because the underlying classifier could be drawn from a variety of models,
 * and ideally, we let the user pick
 */
public abstract class PoseAnnotator {

    /** Tag for the {@link Log}. */
    private static final String TAG = "AnimeRealPoseAnnotator";

    /** Dimensions of inputs. */
    private static final int DIM_BATCH_SIZE = 1;

    private static final int DIM_PIXEL_SIZE = 3;

    /** Preallocated buffers for storing image data in. */
    private int[] intValues = new int[getImageSizeX() * getImageSizeY()];

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter tflite;

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs.
     *  Permanently kept as part of the state of the PoseAnnotator, because
     *  PoseAnnotators are used for managing a single instance of a tensorflow interpreter. */
    protected ByteBuffer imgData = null;


    /** Initializes an {@code ImageClassifier}. */
    PoseAnnotator(Activity activity) throws IOException {
        Log.d(TAG, "model loading");
        tfliteModel = loadModelFile(activity);
        Log.d(TAG, "model Interpreter loading");
        try {
            tflite = new Interpreter(tfliteModel, tfliteOptions);
        }
        catch (Exception e) {
            Log.d(TAG, e.getMessage());
        }

        int in_tensors = tflite.getInputTensorCount();
        int out_tensors = tflite.getOutputTensorCount();
        String in_type = tflite.getInputTensor(0).dataType().toString();
        String out_type = tflite.getOutputTensor(0).dataType().toString();
        String in_shape = Integer.toString(tflite.getInputTensor(0).shape().length);
        String out_shape = Integer.toString(tflite.getOutputTensor(0).shape().length);

        Log.d(TAG, "model Input Type: " + in_type + " Out Type: " + out_type);
        Log.d(TAG, "model Input Shape: " + in_shape + " Out Shape: " + out_shape);


        imgData =
                ByteBuffer.allocateDirect(
                        DIM_BATCH_SIZE
                                * getImageSizeX()
                                * getImageSizeY()
                                * DIM_PIXEL_SIZE
                                * getNumBytesPerChannel());
        imgData.order(ByteOrder.nativeOrder());
        Log.d(TAG, "Created a Tensorflow Lite Pose Annotator.");
    }

    /** Annotates a frame from the preview stream. */
    public AnnotatorOutput annotateFrame(AnnotatorInput input) {
        if (tflite == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.");
            //builder.append(new SpannableString("Uninitialized Classifier."));
        }
        if (input == null) {
            return null;
        }
        convertBitmapToByteBuffer(input.getBitmap());
        // Here's where the magic happens!!!
        long startTime = SystemClock.uptimeMillis();
        AnnotatorOutput result = runInference(input);
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));
        return result;

    }

    private void recreateInterpreter() {
        if (tflite != null) {
            tflite.close();
            // TODO(b/120679982)
            // gpuDelegate.close();
            tflite = new Interpreter(tfliteModel, tfliteOptions);
        }
    }

    public void useCPU() {
        tfliteOptions.setUseNNAPI(false);
        recreateInterpreter();
    }

    public void useNNAPI() {
        tfliteOptions.setUseNNAPI(true);
        recreateInterpreter();
    }

    public void setNumThreads(int numThreads) {
        tfliteOptions.setNumThreads(numThreads);
        recreateInterpreter();
    }

    /** Closes tflite to release resources. */
    public void close() {
        tflite.close();
        tflite = null;
        tfliteModel = null;
    }

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /** Writes Image data into a {@code ByteBuffer}. */
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < getImageSizeX(); ++i) {
            for (int j = 0; j < getImageSizeY(); ++j) {
                final int val = intValues[pixel++];
                addPixelValue(val);
            }
        }
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    }


    /**
     * Get the name of the model file stored in Assets.
     *
     * @return
     */
    protected abstract String getModelPath();

    /*
    Gets the currently inferred probabilistic segmentation of people
    in the scene, where each value is between 0.0 and 1.0
     */
    protected abstract float[][] getPersonSegmentationMap();

    /**
     * Get the image size along the x axis.
     *
     * @return
     */
    protected abstract int getImageSizeX();

    /**
     * Get the image size along the y axis.
     *
     * @return
     */
    protected abstract int getImageSizeY();


    /**
     * Get the image size along the x axis.
     *
     * @return
     */
    protected abstract int getOutputSizeX();

    /**
     * Get the image size along the y axis.
     *
     * @return
     */
    protected abstract int getOutputSizeY();

    /**
     * Get the number of bytes that is used to store a single color channel value.
     *
     * @return
     */
    protected abstract int getNumBytesPerChannel();

    /**
     * Add pixelValue to byteBuffer.
     *
     * @param pixelValue
     */
    protected abstract void addPixelValue(int pixelValue);

    /**
     * Run inference using the prepared input in {@link #imgData}. Afterwards, the result will be
     * provided by getProbability().
     *
     * <p>This additional method is necessary, because we don't have a common base for different
     * primitive data types.
     */
    protected abstract AnnotatorOutput runInference(AnnotatorInput input);

}
