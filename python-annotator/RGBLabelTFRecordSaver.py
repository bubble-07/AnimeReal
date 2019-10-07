#Script which, using a trained depth image annotator
#saved under ./DepthToCoord, annotates all sequences
#with roots immediately under the first argument directory
#and outputs them for RGB image training as packed .tfrecords
#in the given destination directory

import sys
import numpy as np
from RootDirector import *
from RGBTrainingTFWriter import *
import tensorflow as tf
import StandardBody
from multiprocessing import Pool
import traceback
import gc

NUM_PROCS=6

#Given a mask, and a template coordinate matrix in
#depth image index space, returns a template index image
#defined everywhere within the point cloud's (depth coord) index mask
def getTemplateIndexImageFromCoords(mask, coordMatrix):
    coordMask = mask
    flatCoordMask = coordMask.flatten()
    coordFlatInds = np.where(flatCoordMask > 0)

    flatCoordMatrix = np.reshape(coordMatrix, [-1, 3])

    result = np.full((424, 512), StandardBody.pointArray.shape[0], dtype=np.uint16)
    flatResult = result.flatten()

    usedCoords = flatCoordMatrix[coordFlatInds]
    #Great. Now, for the collection of used coordinates, query the Standard Body
    _, labelInds = StandardBody.standardKdTree.query(usedCoords)
    labelInds = labelInds.astype(np.uint16)

    flatResult[coordFlatInds] = labelInds

    return np.reshape(flatResult, (424, 512))
    
def handleSequence(frameManager, destRoot):
    if (not os.path.isdir(destRoot)):
        os.makedirs(destRoot)

    writer = RGBTrainingTFWriter(destRoot)

    tf.reset_default_graph()
    gc.collect()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options)
    #config = tf.ConfigProto(device_count = {'GPU' : 0})

    with tf.Session(config=config) as sess:
        namePrefix = "DepthToCoord"
        saver = tf.train.import_meta_graph('./' + namePrefix + '/' + namePrefix + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./' + namePrefix + '/'))

        graph = tf.get_default_graph()
        #Okay, great. Now what we want are the input/ouput tensor positions

        #Shape: batch_size, 512, 512, 1
        net_eval_in = graph.get_tensor_by_name(namePrefix + "In:0")
        #Shape: batch_size, 512, 512, 3
        net_eval_out = graph.get_tensor_by_name(namePrefix + "Out:0")
        batch_size = 20

        #Y crop to account for 424 vs 512 difference
        lowY = 44
        highY = 468

        done = False

        while (not done):
            in_tensor = np.zeros((batch_size, 512, 512), dtype=np.float32)
            frames = []
            for i in range(batch_size):
                origDepthFrame = frameManager.getFrame().getDepth()
                frames.append(frameManager.getFrame())

                in_tensor[i, lowY:highY, :] = origDepthFrame

                advanced = frameManager.tryAdvance(1)
                if (not advanced):
                    done = True
                    break

            in_tensor = np.reshape(in_tensor, (batch_size, 512, 512, 1))
            feed_dict = {net_eval_in : in_tensor}
            out_tensor = sess.run(net_eval_out, feed_dict)

            for i in range(len(frames)):
                frame = frames[i]
                templatePosOut = out_tensor[i, lowY:highY, :, :]
                rgbImage, mask = frame.getDepthRegisteredRGBAndMask()
                templateIndsOut = getTemplateIndexImageFromCoords(mask, templatePosOut)
                writer.add(rgbImage, templateIndsOut)
        writer.flush()

def handleWorkUnit(argTuple):
    try:
        rootDirector, seq, cam, destPath = argTuple

        frameManager = None
        try:
            frameManager = rootDirector.getFrameManager(seq, cam)
        except IOError:
            print "Missing seq %s and cam %s", seq, cam
            return
        handleSequence(frameManager, destPath)
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))
       

if __name__ == '__main__':
    sequenceRootRoot = sys.argv[1]
    destRoot = sys.argv[2]

    #A collection of work units, defined by the (root director, sequence name, camera name, dest directory)
    workUnits = []

    #Go through every subdirectory of the sequenceRootRoot, and build a rootDirector
    #there for working through the sequences there
    for sub in os.listdir(sequenceRootRoot):
        sequenceRoot = os.path.join(sequenceRootRoot, sub)
        rootDirector = RootDirector(sequenceRoot, init_calibrator=False, init_calib_frames=False, init_background_frames=False)
        #Now, for the root directory, go through every non-empty, non-calib sequence
        for seq in rootDirector.getPeopleRecordingNames():
            for cam in rootDirector.getCameraLabels():
                destPath = os.path.join(os.path.join(os.path.join(destRoot, sub), seq), cam)

                workUnits.append((rootDirector, seq, cam, destPath))

    gc.collect()
    #for workUnit in workUnits:
    #    handleWorkUnit(workUnit)
    #Need max tasks per child to be 1 to be able to free resources!
    p = Pool(NUM_PROCS, maxtasksperchild=1)
    p.map(handleWorkUnit, workUnits)


