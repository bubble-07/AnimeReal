package com.funktorreactive.animereal;

import com.google.ar.sceneform.collision.Ray;
import com.google.ar.sceneform.math.Vector3;

/*
A representation of the data associated with a given point in screen-space, as given by
the underlying neural net model applied to the frame
 */
public class ScreenPointAnnotation {
    Ray screenRay = null;
    Vector3 estimatedTemplatePoint = null;
    Vector3 estimatedWorldPoint = null;
    float templateConfidence;
    float worldConfidence;

    public ScreenPointAnnotation(Ray screenRay, Vector3 estimatedWorldPoint, float worldConfidence, Vector3 templatePoint, float templateConfidence) {
        this.screenRay = screenRay;
        this.estimatedTemplatePoint = templatePoint;
        this.estimatedWorldPoint = estimatedWorldPoint;
        this.worldConfidence = worldConfidence;
        this.templateConfidence = templateConfidence;
    }

    public Vector3 getEstimatedTemplatePoint() {
        return this.estimatedTemplatePoint;
    }

}
