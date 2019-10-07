package com.funktorreactive.animereal;

import android.graphics.ColorSpace;
import android.util.Log;

import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.NodeParent;
import com.google.ar.sceneform.SkeletonNode;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.Renderable;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class ArmatureUtils {

    /*
    Zeroes ALL (local, global) rotation quaternions corresponding to the given skeletonnode
     */
    public static void zeroRotationTree(SkeletonNode character, ModelRenderable characterRenderable) {
        int numBones = characterRenderable.getBoneCount();
        for (int i = 0; i < numBones; i++) {
            String boneName = characterRenderable.getBoneName(i);
            Node boneNode = character.getBoneAttachment(boneName);
            if (boneNode != null) {
                boneNode.setLocalRotation(new Quaternion());
            }
        }
        Node hipsNode = character.getBoneAttachment("Hips");
        if (hipsNode != null) {
            hipsNode.setWorldRotation(new Quaternion());
        }


    }

    public static void rigVRChatArmatureNodes(SkeletonNode character, ModelRenderable characterRenderable) {
        int rootPos = -1;

        int numBones = characterRenderable.getBoneCount();

        //Collection of references to child nodes
        List<Integer>[] childrenInds = new List[numBones];
        Node[] resultNodes = new Node[numBones];

        for (int i = 0; i < numBones; i++) {
            childrenInds[i] = new ArrayList<>();
            resultNodes[i] = null;
        }

        for (int i = 0; i < numBones; i++) {
            int parentNode = characterRenderable.getBoneParentIndex(i);
            String nodeName = characterRenderable.getBoneName(i);
            if (parentNode == -1 && "Armature".equals(nodeName)) {
                //This is the root! Write that down!
                rootPos = i;
            }
            else if (parentNode != -1) {
                //Otherwise, mark this as a child of the parent
                childrenInds[parentNode].add(i);
            }
        }

        //Process nodes depth-first
        Stack<Integer> toProcess = new Stack<>();
        toProcess.push(rootPos);

        while (!toProcess.empty()) {

            int current = toProcess.pop();

            //Log.e("ArmatureUtils", Integer.toString(current));

            int parent = characterRenderable.getBoneParentIndex(current);
            String boneName = characterRenderable.getBoneName(current);

            //Log.e("ArmatureUtils", boneName);


            NodeParent ancestor = null;
            if (parent == -1) {
                ancestor = character;
            }
            else {
                ancestor = resultNodes[parent];
            }
            Node resultNode = new Node();
            resultNode.setParent(ancestor);
            character.setBoneAttachment(boneName, resultNode);

            resultNodes[current] = resultNode;

            //Add children to stacc
            for (Integer childIndex : childrenInds[current]) {
                toProcess.push(childIndex);
            }
        }
    }

}
