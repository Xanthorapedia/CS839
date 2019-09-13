#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

#include <iostream>

PXR_NAMESPACE_USING_DIRECTIVE

class Updatable
{
public:
	virtual void update(float dt) = 0;
};

class Blob : public Updatable
{

public:
	GfVec3f x, v, a, f;
	float m;
	Blob(float m, const GfVec3f &x, const GfVec3f &v = GfVec3f(0), const GfVec3f &a = GfVec3f(0))
		: x(x), v(v), a(a), m(m) {}

	void applyForce(const GfVec3f &f)
	{
		a += f / m;
	};

	void applyForces(const std::vector<GfVec3f> &fs)
	{
		GfVec3f fTotal(0, 0, 0);
		for (const auto &f : fs)
			fTotal += f;
		applyForce(fTotal);
	}

	void update(float dt)
	{
		x += v * dt + 0.5 * a * dt * dt;
		v += a * dt;
		// clear accumulated acceleration
		a.Set(0, 0, 0);
	}

	GfVec3f relPos(const Blob &another) const
	{
		return x - another.x;
	};

	float dist(const Blob &another) const
	{
		return relPos(another).GetLength();
	}

	friend std::ostream &operator<<(std::ostream &strm, const Blob &blob)
	{
		return strm << "Blob(m(" << blob.m << "), x" << blob.x << ", v" << blob.v << ", a" << blob.a << ")";
	}
};

class ElasticRel : public Updatable
{
private:
	Blob &a, &b;
	float dist;
	float stiffness, damping;

public:
	ElasticRel(Blob &a, Blob &b, float stiffness, float damping, float dist)
		: a(a), b(b), damping(damping), stiffness(stiffness), dist(dist) {}

	ElasticRel(Blob &a, Blob &b, float stiffness, float damping)
		: ElasticRel(a, b, stiffness, damping, a.dist(b)) {}

	ElasticRel(Blob &a, Blob &b, float stiffness)
		: ElasticRel(a, b, stiffness, 0) {}

	void update(float dt)
	{
		// unit vector of position from B to A
		GfVec3f uXB2A = b.relPos(a).GetNormalized();
		// damped elastic force of B on A
		GfVec3f feB2A = uXB2A * (b.dist(a) - dist) * stiffness - (a.v - b.v).GetProjection(uXB2A) * damping;
		a.applyForce(feB2A);
		b.applyForce(-feB2A);
	}
};

class Tetroid : public Updatable
{
private:
	std::array<Blob, 4> &blobs;
	std::array<ElasticRel, 6> rels;

public:
	Tetroid(std::array<Blob, 4> &blobs, float stiffness, float damping)
		: blobs(blobs),
		  rels({ElasticRel(blobs[0], blobs[1], stiffness, damping),
				ElasticRel(blobs[0], blobs[2], stiffness, damping),
				ElasticRel(blobs[0], blobs[3], stiffness, damping),
				ElasticRel(blobs[1], blobs[2], stiffness, damping),
				ElasticRel(blobs[1], blobs[3], stiffness, damping),
				ElasticRel(blobs[2], blobs[3], stiffness, damping)}) {}

	void addToMesh(VtVec3fArray &vertices, VtIntArray &faceVertexCounts, VtIntArray &faceVertexIndices)
	{
		int offset = vertices.size();
		for (const auto &b : blobs)
			vertices.push_back(b.x);
		for (int i = 0; i < 4; i++)
			faceVertexCounts.push_back(3);
		for (const auto &idx : {0, 2, 1, 1, 2, 3, 0, 3, 2, 0, 1, 3})
			faceVertexIndices.push_back(offset + idx);
	}

	void updateRels(float dt)
	{
		for (auto &r : rels)
			r.update(dt);
	}

	void update(float dt)
	{
		updateRels(dt);
		for (auto &b : blobs)
			b.update(dt);
	}

	void setVertices(VtVec3fArray &vertices)
	{
		for (int i = 0; i < 4; i++) {
			vertices[i] = blobs[i].x;
		}
	}
};
