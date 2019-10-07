package com.funktorreactive.animereal;

import android.util.Log;

import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.collision.Ray;
import com.google.ar.sceneform.math.Vector3;

import java.util.ArrayList;
import java.util.List;

public class AnnotatorOutput {
    ObservationFrame originalFrame;
    float[][] segmentationMap;

    //Note: this is stored in __normalized__ template body position coordinates.
    //You will need to convert to template coordinates using StandardBody
    Vector3[][] templateMap;

    Ray[][] rayMap;

    public static class SingleObservation {
        private float segmentationProb;
        private Vector3 templatePos;
        private Ray ray;
        public SingleObservation(float segmentationProb, Vector3 normalizedTemplatePos, Ray ray) {
            this.segmentationProb = segmentationProb;
            this.templatePos = normalizedTemplatePos;
            this.ray = ray;
        }
        public Vector3 getNormalizedTemplatePos() {
            return this.templatePos;
        }
        public Ray getRay() {
            return this.ray;
        }
        public float getSegmentationProb() {
            return this.segmentationProb;
        }
    }

    int width;
    int height;

    public SingleObservation getObservationAtIndex(int i, int j) {
        return new SingleObservation(segmentationMap[i][j], templateMap[i][j], rayMap[i][j]);
    }

    public AnnotatorOutput(ObservationFrame originalFrame, byte[][][][] rawOutput) {
        height = rawOutput[0].length;
        width = rawOutput[0][0].length;

        this.segmentationMap = new float[height][width];
        this.templateMap = new Vector3[height][width];
        this.rayMap = new Ray[height][width];

        this.originalFrame = originalFrame;

        initSegmentationMap(rawOutput);
        initTemplateMap(rawOutput);
        initRayMap();
    }

    public BodyNet<List<SingleObservation>> getBodyNet(StandardBody body, float minConf) {
        BodyNet<List<SingleObservation>> result = new BodyNet<>(body, () -> new ArrayList<>());
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                SingleObservation observation = getObservationAtIndex(i, j);
                if (observation.getSegmentationProb() > minConf) {
                    List<SingleObservation> L = result.getDataAtNormalized(observation.getNormalizedTemplatePos());
                    L.add(observation);
                }
            }
        }
        return result;
    }

    public TimeStamp getTime() {
        return this.originalFrame.getTime();
    }

    private void initRayMap() {
        for (int i = 0; i < height; i++) {
            float yFrac = ((float) i) / ((float) height);
            for (int j = 0; j < width; j++) {
                float xFrac = ((float) j) / ((float) width);
                rayMap[i][j] = originalFrame.getFrameRay(xFrac, yFrac);
            }
        }
    }

    private void initSegmentationMap(byte[][][][] poseAnnotatorResults) {
        this.segmentationMap = new float[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                segmentationMap[i][j] = ((float) poseAnnotatorResults[0][i][j][3]) / (255.0f);
            }
        }
    }

    private Vector3 convertTemplatePos(byte[] raw) {
        float[] temp = new float[3];
        for (int i = 0; i < 3; i++) {
            temp[i] = ((float) raw[i]) / 255.0f;
        }
        return new Vector3(temp[0], temp[1], temp[2]);
    }

    private void initTemplateMap(byte[][][][] raw) {
        this.templateMap = new Vector3[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                byte[] byteValues = raw[0][i][j];
                Vector3 temp = convertTemplatePos(byteValues);
                this.templateMap[i][j] = temp;
            }
        }
    }
}
