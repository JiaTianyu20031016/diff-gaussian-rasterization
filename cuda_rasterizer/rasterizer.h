/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* features,
			const float* features_deltaS,
			const float* features_deltaR,
			const float* features_deltaX,
			float* deltaSs,
			float* deltaRs,
			float* deltaXs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,										// keep in mind: deformation should be included
			const float* deltaXs,
			const float* features,
			const float* features_deltaS,
			const float* features_deltaR,
			const float* features_deltaX,
			const float* colors_precomp,
			const float* scales,										// keep in mind: deformation should be included, but not activated
			const float scale_modifier,
			const float* rotations,										// keep in mind: deformation should be included, but not activated
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* img_buffer,
			const float* dL_dpix,
			const float* dL_ddeltaS,
			const float* dL_ddeltaR,
			const float* dL_ddeltaX,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dfeature,
			float* dL_dfeature_deltaS,
			float* dL_dfeature_deltaR,
			float* dL_dfeature_deltaX,
			float* dL_dscale,
			float* dL_drot,
			bool debug);
	};
};

#endif