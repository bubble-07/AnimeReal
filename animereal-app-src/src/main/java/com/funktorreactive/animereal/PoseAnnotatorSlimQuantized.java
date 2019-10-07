package com.funktorreactive.animereal;

import android.app.Activity;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.text.style.ForegroundColorSpan;

import java.io.IOException;

public class PoseAnnotatorSlimQuantized extends PoseAnnotator {
    /**
     * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
     * of the super class, because we need a primitive array here.
     */
    private byte[][][][] poseAnnotatorResults = null;

    /**
     * Initializes an {@code PoseAnnotatorSlim}.
     *
     * @param activity
     */
    PoseAnnotatorSlimQuantized(Activity activity) throws IOException {
        super(activity);
        poseAnnotatorResults = new byte[1][getOutputSizeY()][getOutputSizeX()][4];
    }

    @Override
    protected String getModelPath() {
        // you can download this file from
        // see build.gradle for where to obtain this file. It should be auto
        // downloaded into assets.
        return "pose_annotator_slim.tflite";
    }

    @Override
    protected float[][] getPersonSegmentationMap() {
        float[][] segMap = new float[getOutputSizeY()][getOutputSizeX()];
        for (int i = 0; i < getOutputSizeY(); i++) {
            for (int j = 0; j < getOutputSizeX(); j++) {
                segMap[i][j] = ((float) poseAnnotatorResults[0][i][j][3]) / (255.0f);
            }
        }
        return segMap;
    }

    @Override
    protected int getOutputSizeX() {
        return 64;
    }

    @Override
    protected int getOutputSizeY() {
        return 64;
    }

    @Override
    protected int getImageSizeX() {
        return 256;
    }

    @Override
    protected int getImageSizeY() {
        return 256;
    }

    @Override
    protected int getNumBytesPerChannel() {
        return 1;
    }

    @Override
    protected void addPixelValue(int pixelValue) {
        imgData.put((byte) ((pixelValue >> 16) & 0xFF));
        imgData.put((byte) ((pixelValue >> 8) & 0xFF));
        imgData.put((byte) (pixelValue & 0xFF));
    }


    @Override
    protected AnnotatorOutput runInference(AnnotatorInput input) {
        tflite.run(imgData, poseAnnotatorResults);
        ObservationFrame originalFrame = input.getObservationFrame();
        return new AnnotatorOutput(originalFrame, this.poseAnnotatorResults);
    }

}
