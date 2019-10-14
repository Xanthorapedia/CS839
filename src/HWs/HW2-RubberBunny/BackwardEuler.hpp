#pragma once

#include "cuCG.hpp"

namespace BackwardEuler
{

// error threshold
const float eps = 1e-4f;

using Vec3 = Eigen::Vector3f;
using Mat3 = Eigen::Matrix3f;
using ConstraintFunc = std::vector<unsigned> (*)(cuCG::SimCfg &scfg);

template <int N_TETR>
class Solver
{
private:
	cuCG::SimCfg *scfg;
	ConstraintFunc imposeConstraint;

	void createCfg(const size_t N_PART, const size_t N_CELL)
	{
		// initialize config struct
		HANDLE_ERROR(cudaHostAlloc((void **)&scfg, sizeof(cuCG::SimCfg), cudaHostAllocMapped));
		HANDLE_ERROR(cudaMalloc((void **)&scfg->tetrIdcs, N_CELL * 4 * sizeof(unsigned)));
		HANDLE_ERROR(cudaMalloc((void **)&scfg->tetrKind, N_CELL * sizeof(unsigned)));
		HANDLE_ERROR(cudaMalloc((void **)&scfg->dmInvs, N_CELL * sizeof(Mat3)));
		HANDLE_ERROR(cudaMalloc((void **)&scfg->restVs, N_CELL * sizeof(float)));
		HANDLE_ERROR(cudaMallocManaged((void **)&scfg->m, N_PART * sizeof(Vec3)));
		HANDLE_ERROR(cudaMallocManaged((void **)&scfg->x, N_PART * sizeof(Vec3)));
		HANDLE_ERROR(cudaMallocManaged((void **)&scfg->v, N_PART * sizeof(Vec3)));

		// cg
		HANDLE_ERROR(cudaMalloc((void **)&scfg->dx, N_PART * sizeof(Vec3)));
		HANDLE_ERROR(cudaMalloc((void **)&scfg->b, N_PART * sizeof(Vec3)));
		HANDLE_ERROR(cudaMalloc((void **)&scfg->p, N_PART * sizeof(Vec3)));
		HANDLE_ERROR(cudaMalloc((void **)&scfg->q, N_PART * sizeof(Vec3)));
		HANDLE_ERROR(cudaMalloc((void **)&scfg->r, N_PART * sizeof(Vec3)));
		HANDLE_ERROR(cudaMallocManaged((void **)&scfg->scratch_arr, N_PART * sizeof(float)));

		// ceil(N_PART / sizeof(unsigned))
		unsigned N_PART_align32 = (N_PART + sizeof(unsigned)) / sizeof(unsigned);
		HANDLE_ERROR(cudaMallocManaged((void **)&scfg->mutex, N_PART_align32 * sizeof(unsigned)));

		scfg->constrained = NULL;
	}

public:
	Solver(){};
	void init(std::vector<Vec3> &particleX,
			  std::vector<std::array<unsigned, 5>> &tetrInfo,
			  std::vector<float> m,
			  std::array<Mat3, N_TETR> &tetrDmInvs,
			  std::array<float, N_TETR> &tetrRestVs,
			  float mu, float lambda, float gamma,
			  ConstraintFunc imposeConstraintFunc = NULL)
	{
		const size_t N_PART = particleX.size();
		const size_t N_CELL = tetrInfo.size();

		// malloc stuff
		createCfg(N_PART, N_CELL);

		std::vector<unsigned> tetrIdces, tetrKinds;
		for (auto &&tetrInf : tetrInfo)
		{
			tetrIdces.insert(tetrIdces.end(), tetrInf.begin(), tetrInf.end() - 1);
			tetrKinds.push_back(tetrInf[4]);
		}

		// populate data
		scfg->N_CELL = N_CELL;
		scfg->N_PART = N_PART;
		scfg->N_TETR = N_TETR;

		std::vector<Vec3> zeros(N_PART);
		std::fill(zeros.begin(), zeros.end(), Vec3::Zero());

		HANDLE_ERROR(cudaMemcpy(scfg->tetrIdcs, tetrIdces.data(),
								N_CELL * 4 * sizeof(unsigned), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(scfg->tetrKind, tetrKinds.data(),
								N_CELL * sizeof(unsigned), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(scfg->m, m.data(),
								N_PART * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(scfg->x, particleX.data(),
								N_PART * sizeof(Vec3), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(scfg->v, zeros.data(),
								N_PART * sizeof(Vec3), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(scfg->dmInvs, tetrDmInvs.data(),
								N_TETR * sizeof(Mat3), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(scfg->restVs, tetrRestVs.data(),
								N_TETR * sizeof(float), cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaMemcpy(scfg->b, zeros.data(),
								N_PART * sizeof(Vec3), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(scfg->p, zeros.data(),
								N_PART * sizeof(Vec3), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(scfg->q, zeros.data(),
								N_PART * sizeof(Vec3), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(scfg->r, zeros.data(),
								N_PART * sizeof(Vec3), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(scfg->dx, zeros.data(),
								N_PART * sizeof(Vec3), cudaMemcpyHostToDevice));

		// ceil(N_PART / sizeof(unsigned))
		unsigned N_PART_align32 = (N_PART + sizeof(unsigned)) / sizeof(unsigned);
		HANDLE_ERROR(cudaMemset(scfg->mutex, 0, N_PART_align32 * sizeof(unsigned)));

		scfg->t = 0;
		scfg->mu = mu;
		scfg->lambda = lambda;
		scfg->gamma = gamma;
		HANDLE_ERROR(cudaDeviceSynchronize());

		imposeConstraint = imposeConstraintFunc;
	}

	void update(float dt)
	{
		scfg->dt = dt;
		scfg->t += dt;

		HANDLE_ERROR(cudaDeviceSynchronize());

		if (imposeConstraint)
		{
			std::vector<unsigned> toKeep = imposeConstraint(*scfg);

			// needs resize
			if (toKeep.size() > scfg->N_CONSTR)
			{
				if (scfg->constrained)
					HANDLE_ERROR(cudaFree(scfg->constrained));
				HANDLE_ERROR(cudaMalloc((void **)&scfg->constrained, toKeep.size() * sizeof(unsigned)));
			}

			// copy
			HANDLE_ERROR(cudaMemcpy(scfg->constrained, toKeep.data(),
									toKeep.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
			scfg->N_CONSTR = toKeep.size();
		}

		cuCG::progress(scfg, eps);

		// make sure everything is done before reading from gpu
		HANDLE_ERROR(cudaDeviceSynchronize());
	}

	void getRslt(std::vector<Vec3> &particleX)
	{
		std::copy(scfg->x, scfg->x + scfg->N_PART, particleX.begin());
	}

	~Solver()
	{
		HANDLE_ERROR(cudaFree(scfg->tetrIdcs));
		HANDLE_ERROR(cudaFree(scfg->tetrKind));
		HANDLE_ERROR(cudaFree(scfg->dmInvs));
		HANDLE_ERROR(cudaFree(scfg->restVs));
		HANDLE_ERROR(cudaFree(scfg->m));
		HANDLE_ERROR(cudaFree(scfg->x));
		HANDLE_ERROR(cudaFree(scfg->v));

		HANDLE_ERROR(cudaFree(scfg->constrained));

		HANDLE_ERROR(cudaFree(scfg->dx));
		HANDLE_ERROR(cudaFree(scfg->b));
		HANDLE_ERROR(cudaFree(scfg->p));
		HANDLE_ERROR(cudaFree(scfg->q));
		HANDLE_ERROR(cudaFree(scfg->r));
		HANDLE_ERROR(cudaFree(scfg->scratch_arr));

		HANDLE_ERROR(cudaFree(scfg->mutex));

		HANDLE_ERROR(cudaFreeHost(scfg));
	}
};

} // namespace BackwardEuler
