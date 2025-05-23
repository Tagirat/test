#include <cmath>
#include <fstream>
#include <iostream>

struct Point {
  double x, y;
};

struct Circle {
  Point center;
  double radius;
};

int getPosition(const Point *p, const Circle *c) {
  double dx = p->x - c->center.x;
  double dy = p->y - c->center.y;
  double epsilon = 1e-9;

  if (std::fabs(dx * dx + dy * dy - c->radius * c->radius) < epsilon)
    return 0;
  return (dx * dx + dy * dy < c->radius * c->radius) ? 1 : 2;
}

int main() {
  std::ifstream circle_file("file1.txt");
  if (!circle_file) {
    std::cerr << "Error of opening file1.txt" << std::endl;
    return 1;
  }

  Circle circle;
  circle_file >> circle.center.x >> circle.center.y >> circle.radius;
  std::ifstream points_file("file2.txt");
  if (!points_file) {
    std::cerr << "Error of oppening file2.txt" << std::endl;
    return 1;
  }

  int count = 0;
  double x, y;
  while (points_file >> x >> y) {
    count++;
  }

  points_file.clear();
  points_file.seekg(0);

  Point *points = new Point[count];
  for (int i = 0; i < count; i++) {
    points_file >> points[i].x >> points[i].y;
  }
  points_file.close();

  for (int i = 0; i < count; i++) {
    std::cout << getPosition(&points[i], &circle) << std::endl;
  }

  delete[] points;
  return 0;
}
