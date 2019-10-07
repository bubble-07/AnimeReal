package com.funktorreactive.animereal;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.Resources;
import android.util.Log;

import com.google.ar.sceneform.math.Vector3;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

import de.biomedical_imaging.edu.wlu.cs.levy.CG.KDTree;
import de.biomedical_imaging.edu.wlu.cs.levy.CG.KeyDuplicateException;
import de.biomedical_imaging.edu.wlu.cs.levy.CG.KeySizeException;

/*
Provides information and utilities related to (a cover of) the standard template body.
There should only ever be one instance of this class, but it also shouldn't
matter (other than for performance) if more than one is instantiated
 */
public class StandardBody {
    private double[][] pointArray = null;
    private KDTree<Integer> indexTree = null;
    private int[][] neighbors = null;
    //Number of neighbors to track for each node, including the point itself
    private int NUM_NEIGHBORS = 4;

    double[] minBounds = {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY};
    double[] maxBounds = {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY};

    public int getNumPoints() {
        return pointArray.length;
    }
    public int getNumNeighbors() {
        return neighbors.length;
    }

    public int getNearestIndexToNormalized(Vector3 point) {
        return getNearestIndex(normalizedToTemplateCoords(point));
    }

    public int getNearestIndex(Vector3 point) {
        double[] key = {(double) point.x, (double) point.y, (double) point.z};
        try {
            return this.indexTree.nearest(key);
        }
        catch (KeySizeException e) {
            return -1;
        }
    }

    /*
    Convert from template-normalized coords (between 0.0 and 1.0)
    to template coords
     */
    public Vector3 normalizedToTemplateCoords(Vector3 normalized) {
        double[] input = new double[]{(double)normalized.x, (double)normalized.y, (double)normalized.z};
        for (int i = 0; i < input.length; i++) {
            input[i] = minBounds[i] + (maxBounds[i] - minBounds[i]) * input[i];
        }
        return new Vector3((float)input[0], (float)input[1], (float)input[2]);

    }

    public Vector3 getPointAtIndex(int ind) {
        double[] point = pointArray[ind];
        return new Vector3((float) point[0], (float) point[1], (float) point[2]);
    }

    public Vector3 getNearestPoint(Vector3 point) {
        return getPointAtIndex(getNearestIndex(point));
    }

    public List<Integer> getNearestIndices(Vector3 point, int n) {
        double[] key = {(double) point.x, (double) point.y, (double) point.z};
        try {
            return this.indexTree.nearest(key, n);
        }
        catch (KeySizeException e) {
            return new ArrayList<>();
        }
    }

    public int[] getNeighborIndices(int ind) {
        return this.neighbors[ind];
    }

    public StandardBody(Activity activity) {

        try {

            AssetFileDescriptor fileDescriptor = activity.getAssets().openFd("cover50.csv");
            this.initPointArray(fileDescriptor.createInputStream());
        }
        catch (IOException e) {
            Log.e("StandardBody", "FATAL DURING LOADING: " + e.getMessage());
        }
        initIndexTree();
        initNeighbors();
    }

    private void initPointArray(FileInputStream is) throws IOException {
        CSVParser parser = CSVParser.parse(is, Charset.defaultCharset(), CSVFormat.DEFAULT);
        List<CSVRecord> recordList = parser.getRecords();
        pointArray = new double[recordList.size()][3];
        for (int i = 0; i < recordList.size(); i++) {
            CSVRecord record = recordList.get(i);
            for (int j = 0; j < 3; j++) {
                String valStr = record.get(j);
                double val = Double.parseDouble(valStr);

                if (val > maxBounds[j]) {
                    maxBounds[j] = val;
                }
                if (val < minBounds[j]) {
                    minBounds[j] = val;
                }

                pointArray[i][j] = val;
            }
        }
        is.close();
    }

    private void initIndexTree() {
        this.indexTree = new KDTree<>(3);
        for (int i = 0; i < pointArray.length; i++) {
            try {
                this.indexTree.insert(pointArray[i], i);
            }
            catch (KeyDuplicateException e) {
                Log.e("StandardBody", "FATAL DURING LOADING: " + e.getMessage());

            }
            catch (KeySizeException e) {
                Log.e("StandardBody", "FATAL DURING LOADING: " + e.getMessage());
            }
        }
    }
    private void initNeighbors() {
        this.neighbors = new int[pointArray.length][NUM_NEIGHBORS];
        //Fill the neighbors array
        for (int i = 0; i < pointArray.length; i++) {
            double[] point = pointArray[i];
            List<Integer> neighbors = new ArrayList<>();
            try {
                neighbors = this.indexTree.nearest(point, this.NUM_NEIGHBORS);
            }
            catch (KeySizeException e) {
                Log.e("StandardBody", "FATAL DURING LOADING: " + e.getMessage());
            }
            for (int j = 0; j < this.NUM_NEIGHBORS; j++) {
                this.neighbors[i][j] = neighbors.get(j);
            }
        }
    }
}
