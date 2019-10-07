package com.funktorreactive.animereal;

import com.google.ar.sceneform.FrameTime;

import java.sql.Time;

public class TimeStamp implements Comparable<TimeStamp> {

    float startSeconds;

    public TimeStamp(FrameTime frameTime) {
       this.startSeconds = frameTime.getStartSeconds();
    }

    @Override
    public int compareTo(TimeStamp o) {
        return Float.compare(startSeconds, o.startSeconds);
    }

    public float getStartSeconds() {
        return this.startSeconds;
    }
}
