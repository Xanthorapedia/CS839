#pragma once

#include <Eigen/Dense>
// #include <vector>

// Lifted from:
// https://doc.cgal.org/latest/AABB_tree/index.html

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

using K = CGAL::Simple_cartesian<float>;
using Point = K::Point_3;
using Ray = K::Ray_3;
using Triangle = K::Triangle_3;

using TriangleItr = std::vector<Triangle>::iterator;
using TreePrimitive = CGAL::AABB_triangle_primitive<K, TriangleItr>;
using TriangleTraits = CGAL::AABB_traits<K, TreePrimitive>;
using TriangleTree = CGAL::AABB_tree<TriangleTraits>;

using Vec3 = Eigen::Vector3f;

namespace Lattice
{

class AABBTree
{
private:
	std::vector<Triangle> triangleList;
	TriangleTree tree;

public:
	AABBTree() {}

	std::vector<Triangle> &getTriangleList()
	{
		return triangleList;
	}

	void insert()
	{
		tree.insert(triangleList.begin(), triangleList.end());
		tree.accelerate_distance_queries();
	}

	bool encloses(const Vec3 &point)
	{
		auto b = tree.bbox();
		Point end(b.xmax(), b.ymax(), b.zmax());
		return tree.number_of_intersected_primitives(Ray(Point(point.x(), point.y(), point.z()), end)) % 2;
	}

	float squaredDistance(const Vec3 &point)
	{
		return tree.squared_distance(Point(point.x(), point.y(), point.z()));
	}

	Vec3 vecToSurface(const Vec3 &point, unsigned &fid, unsigned &vid)
	{
		Point q(point.x(), point.y(), point.z());

		auto result = tree.closest_point_and_primitive(q);

		// project onto surface
		Point pq = (*result.second).supporting_plane().projection(q);

		// find which point on which triangle is the closest
		fid = result.second - triangleList.begin();
		for (vid = 0; vid < 3; vid++)
			if ((*result.second).vertex(vid) == result.first)
				break;

		return Vec3(pq.x() - q.x(), pq.y() - q.y(), pq.z() - q.z());
	}
};

} // namespace Lattice