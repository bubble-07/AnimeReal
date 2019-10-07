#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "math.h"
#include <algorithm>

using namespace tensorflow;

//Custom op definition for heatmap generation. Takes in a tensor of
//shape (num_bodies, num_parts, 4) representing annotation keypoints
//and returns a (256, 256, num_parts)
//tensor of heatmaps generated from those annotations, assuming that
//the original image was of size 480x480 and we're looking at the 256x256 central crop

//NOTE: Why do this as a custom op? Because you can do it with broadcasting in native TF,
//but the result is horrible to read and slow as molasses

//The arguments for the op are the annotation tensor (which, recall, is (num_bodies, num_parts, 4)
//of (x, y, z, c) where (x, y) is zero-centered at the center of the image and c is the confidence level)
//the list of per-part scaling factors,
//and two focal length parameters s and t. If the part's scaling
//factor is p, then the Gaussian standard deviation will become:
//(1 / (s z + t)) * p, where z is the z-location of an annotation point
REGISTER_OP("HeatmapGen")
    .Input("annotation_tensor : float32")
    .Input("part_scaling_facs : float32")
    .Input("s : float32")
    .Input("t : float32")
    .Output("heatmap_tensor : float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle input;

        ::tensorflow::shape_inference::DimensionHandle unused;

        auto anno_in = c->input(0);
        auto scaling_in = c->input(1);
        auto s_in = c->input(2);
        auto t_in = c->input(3);

        //RESTRICTIONS ON FIRST INPUT
        //First, ensure that the first input (the annotation tensor)
        //has rank of exactly three
        TF_RETURN_IF_ERROR(c->WithRank(anno_in, 3, &input));

        //Then, ensure that the annotation input's third component has a value of exactly 4
        TF_RETURN_IF_ERROR(c->WithValue(c->Dim(anno_in, 2), 4, &unused));

        //Also determine the number of parts (no constraint enforced, since this could in theory change)
        auto num_parts = c->Dim(anno_in, 1);

        //RESTRICTIONS ON SECOND INPUT
        //Ensure that the second input (part scaling factors) is rank-1
        //and of dimension [num_parts]
        TF_RETURN_IF_ERROR(c->WithRank(scaling_in, 1, &input));
        TF_RETURN_IF_ERROR(c->Merge(c->Dim(scaling_in, 0), num_parts, &unused));

        //RESTRICTIONS ON THIRD AND FOURTH INPUTS
        //Ensure that these are just scalars

        TF_RETURN_IF_ERROR(c->WithRank(s_in, 0, &input));
        TF_RETURN_IF_ERROR(c->WithRank(t_in, 0, &input));

        //Great, now the output size is (256, 256, num_parts)
        c->set_output(0, c->MakeShape({256, 256, num_parts }));

        return Status::OK();
    });

//Helper function which fills an array of size (end - start)
//containing computed values for a bell-shaped height * (e^(-(x - u)^2 / 2*sigma^2) ) curve
//Here, [start, end) is the integer range of x-values to compute the array for
void fillBellCurve(float target[], int start, int end, float u, float sigma, float height) {
    float neg_recip_twosigmasq = -1.0 / (2.0 * (sigma * sigma));

    for (int x = start; x < end; x++) {
        float diff = ((float) x) - u;
        float sqdiff = diff * diff;

        int ind = x - start;

        target[ind] = height * exp(sqdiff * neg_recip_twosigmasq);
    }
}

//TODO: Does this benefit at all from multi-threading?
//TODO: Does this benefit at all from placing it on the GPU?
class HeatmapGenOp : public OpKernel {
    public:
    explicit HeatmapGenOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        //Grab the annotation tensor as an Eigen::Tensor
        const Tensor& input_tensor = context->input(0);

        auto num_bodies = input_tensor.dim_size(0);
        auto num_parts = input_tensor.dim_size(1);

        //Eigen::Tensor for processing
        auto anno_in = input_tensor.tensor<float, 3>();

        //Grab the part scaling factor tensor as an Eigen::Vector
        const Tensor& scaling_tensor = context->input(1);
        auto scaling_in = scaling_tensor.vec<float>();

        //Now, grab scalar focal length factors s and t
        const Tensor& s_tensor = context->input(2);
        const Tensor& t_tensor = context->input(3);
        float s_in = s_tensor.scalar<float>()(0);
        float t_in = t_tensor.scalar<float>()(0);
    
        //Create space for the output
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, 
                                 ::tensorflow::TensorShape({256, 256, num_parts}), &output_tensor));
        //Eigen::Tensor for output
        auto out = output_tensor->tensor<float, 3>();

        //First, zero the output tensor.
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                for (int k = 0; k < num_parts; k++) {
                    out(i, j, k) = 0.0f;
                }
            }
        }

        //Alright, now let's get moving!
        //Loop ordering: loop through each body, then each part, then each pixel in the output (y, x)
        //TODO: For parallelization (if using GPU kernel), would a different ordering be better?
        
        for (int body_ind = 0; body_ind < num_bodies; body_ind++) {
            for (int part_ind = 0; part_ind < num_parts; part_ind++) {
                //Get the annotation coordinates
                //TODO: Is it faster to get slices or something? Don't know much about Eigen::Tensor
                //Here, we add 128 to anno_x and anno_y because the annotation was passed
                //in a centered coordinate system, but for the output, we care about
                //a 256x256 integer-indexed square centered on the origin
                float anno_x = anno_in(body_ind, part_ind, 0) + 128.0f;
                float anno_y = anno_in(body_ind, part_ind, 1) + 128.0f;
                float anno_z = anno_in(body_ind, part_ind, 2);
                float anno_c = anno_in(body_ind, part_ind, 3);

                //Compute the standard deviation for this part, which is (p / (s z + t))
                float std_dev = scaling_in(part_ind) / fmaf(s_in, anno_z, t_in);

                //The number of standard deviations out from the annotated point
                //to care about computing the function
                float stddev_limit = 3.0f;

                float extent = std_dev * stddev_limit;

                //From the annotation, find x and y coordinate limits for a bounding
                //box to compute the bell-shaped curve here
                int low_x = std::max(0, ((int) (anno_x - extent)));
                int high_x = std::min(256, ((int) (anno_x + extent)));

                int low_y = std::max(0, ((int) (anno_y - extent))); 
                int high_y = std::min(256, ((int) (anno_y + extent)));

                int x_extent = high_x - low_x;
                int y_extent = high_y - low_y;

                //If x_extent or y_extent is zero or negative, that must mean we've hit
                //clipping, so just skip this one's contribution entirely
                if (x_extent <= 0 || y_extent <= 0) {
                    continue;
                }


                //Allocate and fill independent bell-curve lookup arrays for
                //x and y. This exploits the fact that (e^-(x^2)) * (e^-(y^2)) = e^(-(x^2 + y^2))

                float x_bell_curve[x_extent];
                float y_bell_curve[y_extent];

                fillBellCurve(x_bell_curve, low_x, high_x, anno_x, std_dev, 1.0);
                //The second bell curve is scaled by the confidence so we don't have to do that later
                fillBellCurve(y_bell_curve, low_y, high_y, anno_y, std_dev, anno_c);

                for (int y_offset = 0; y_offset < y_extent; y_offset++) {
                    int y = low_y + y_offset;
                    float yval = y_bell_curve[y_offset];

                    int x = low_x;
                    for (int x_offset = 0; x_offset < x_extent; x_offset++) {
                        out(y, x, part_ind) += x_bell_curve[x_offset] * yval;
                        x++;
                    }
                }
            }
        }
        //Once we've added all of those contributions, we should be good
    }
};

REGISTER_KERNEL_BUILDER(Name("HeatmapGen").Device(DEVICE_CPU), HeatmapGenOp)
