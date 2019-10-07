package com.funktorreactive.animereal;


import android.graphics.Bitmap;

import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.collision.Ray;

/*
Plain-old-data class which is used to pass information to the annotator
Includes the world-rays corresponding to the four corners of the screen
at the time of capture, the bitmap at the time of capture, and the time
of the capture
 */
public class AnnotatorInput {
    Bitmap captured;
    ObservationFrame observationFrame;
    public AnnotatorInput(ObservationFrame obsFrame, Bitmap captured) {
        this.captured = captured;
        this.observationFrame = obsFrame;
    }

    public Bitmap getBitmap() {
        return this.captured;
    }
    public ObservationFrame getObservationFrame() {
        return this.observationFrame;
    }
    public void recycle() {
        this.captured.recycle();
    }
}
