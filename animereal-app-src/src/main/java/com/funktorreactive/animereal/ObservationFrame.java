package com.funktorreactive.animereal;

import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.SceneView;
import com.google.ar.sceneform.collision.Ray;
import com.google.ar.sceneform.math.Vector3;

/*
Denotes a camera view at a given point in time, by world-rays at the corners
together with the frametime
 */
public class ObservationFrame {
    TimeStamp time;
    Vector3[] directions = new Vector3[4];
    Vector3 cameraPos;

    //Rays should be specified in a _clockwise_ order from 0, 0
    public ObservationFrame(TimeStamp time, Ray[] corners) {
        this.time = time;
        for (int i = 0; i < 4; i++) {
            directions[i] = corners[i].getDirection();
        }
        cameraPos = corners[0].getOrigin();
    }
    public TimeStamp getTime() {
        return this.time;
    }

    public Vector3 upperLeftDirection() {
        return this.directions[0];
    }
    public Vector3 upperRightDirection() {
        return this.directions[1];
    }
    public Vector3 lowerLeftDirection() {
        return this.directions[3];
    }

    public Vector3 right() {
        return Vector3.subtract(upperRightDirection(), upperLeftDirection());
    }

    public Vector3 down() {
        return Vector3.subtract(lowerLeftDirection(), upperLeftDirection());
    }

    public Ray getFrameRay(float xFrac, float yFrac) {
        Vector3 rightX = right().scaled(xFrac);
        Vector3 downY = down().scaled(yFrac);
        Vector3 displacement = Vector3.add(rightX, downY);
        Vector3 upperLeft = upperLeftDirection();

        Vector3 resultDir = Vector3.add(upperLeft, displacement);

        return new Ray(this.cameraPos, resultDir);

    }
}
