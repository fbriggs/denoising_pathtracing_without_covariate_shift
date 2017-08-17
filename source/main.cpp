/*
Copyright 2017 Forrest Briggs

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

I am providing code in this repository to you under an open source license. Because this is my personal repository, the license you receive to my code is from me, and not from my employer (Facebook).
*/

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <cfloat>
#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "HyperDecisionTree.h"

using namespace std;
using namespace hyper_decision_tree;
typedef std::chrono::high_resolution_clock Clock;

struct Vec3f {
  float x, y, z;
  Vec3f() : x(0), y(0), z(0) {}
  Vec3f(const float x, const float y, const float z) : x(x), y(y), z(z)  {}
  Vec3f operator+(const Vec3f &b) const { return Vec3f(x + b.x, y + b.y, z + b.z); }
  Vec3f operator-(const Vec3f &b) const { return Vec3f(x - b.x, y - b.y, z - b.z); }
  Vec3f operator*(const float b) const { return Vec3f(x * b, y * b, z * b); }
  Vec3f operator/(const float b) const { return Vec3f(x / b, y / b, z / b); }
  Vec3f cwiseMult(const Vec3f &b) const { return Vec3f(x * b.x, y * b.y, z * b.z); }
  Vec3f& norm() { return *this = *this * (1.0 / sqrtf(x * x + y * y + z * z)); }
  float dot(const Vec3f &b) const { return x*b.x+y*b.y+z*b.z; }
  Vec3f operator%(const Vec3f& b) const { return Vec3f(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }  // cross
};

struct Sphere {
  Vec3f center;
  float radius;

  Vec3f color; // surface color
  Vec3f emitColor; // color of light emitted by surface
  float specularCoef;
  float specularRadius; // 0 = perfect mirror, 1 = very spread out

  Sphere(
    const Vec3f& center,
    const float radius,
    const Vec3f& color,
    const Vec3f& emitColor,
    const float specularCoef,
    const float specularRadius
  ) :
    center(center),
    radius(radius),
    color(color),
    emitColor(emitColor),
    specularCoef(specularCoef),
    specularRadius(specularRadius)
  {}

  bool intersect(
    const Vec3f& rayOrigin,
    const Vec3f& rayDir,
    float& intersectionDist
  ) const {
    const Vec3f rayToSphereCenter = center - rayOrigin;
    const float lengthRTSC2 = rayToSphereCenter.dot(rayToSphereCenter);

    const float closestApproach = rayToSphereCenter.dot(rayDir);
    if (closestApproach < 0 ) { return false; }

    const float halfCord2 =
      (radius * radius) - lengthRTSC2 + (closestApproach * closestApproach);
    if(halfCord2 < 0) { return false; }

    intersectionDist = closestApproach - sqrtf(halfCord2); // TODO: optimize no sqrt for shadow rays
    return true;
  }
};

inline float randf01() { return float(rand()) / float(RAND_MAX); }

static inline Vec3f randomUnitVec() {
  const float r1 = randf01();
  const float r2 = randf01();
  const float phi = 2.0f * M_PI * r1;
  const float theta = acos(1.0f - 2.0f * r2);
  const float x = 2.0f * cos(2.0f * M_PI * r1) * sqrtf(r2 * (1.0f - r2));
  const float y = 2.0f * sin(2.0f * M_PI * r1) * sqrtf(r2 * (1.0f - r2));
  const float z = 1.0 - 2.0f * r2;
  return Vec3f(x, y, z);
}

static inline Vec3f randomUnitVecCosDistrib(const Vec3f& n) {
  const Vec3f uu = (Vec3f(0, 1, 1) % n).norm();
  const Vec3f vv = uu % n;
  const float r0 = randf01();
  const float r1 = randf01();
  const float ra = sqrtf(r1);
  const float rx = ra * cosf(2.0 * M_PI * r0);
  const float ry = ra * sinf(2.0 * M_PI * r0);
  const float rz = sqrtf(1.0 - r1);
  return (uu * rx + vv * ry + n * rz).norm();
}

static Vec3f directLight(
  const Vec3f& point,
  const Vec3f& normal,
  const vector<Sphere>& spheres,
  const vector<int>& lightIndices
) {
  Vec3f sumLight;
  for (int l : lightIndices) { // TODO: maybe pick only one light at random
    const Vec3f pointOnLight =
      spheres[l].center + randomUnitVec() * spheres[l].radius;
    const Vec3f dirToLight = (pointOnLight - point).norm();
    const float ndl = std::max(0.0f, normal.dot(dirToLight));
    if (ndl <= 0.0) { continue; }

    float closest = DBL_MAX;
    int closestId = -1;
    for (int i = 0; i < spheres.size(); ++i)  {
      float dist;
      if (spheres[i].intersect(point, dirToLight, dist) && dist < closest) {
        closest = dist;
        closestId = i;
      }
    }
    if (closestId == l) {
      sumLight = sumLight + spheres[l].emitColor * ndl;
    }
  }
  return sumLight;
}

static Vec3f pathTrace(
  const Vec3f& rayOrigin,
  const Vec3f& rayDir,
  const vector<Sphere>& spheres,
  const vector<int>& lightIndices,
  const int depth = 0
) {
  static const int kMaxDepth = 3;
  if (depth >= kMaxDepth) { return Vec3f(0,0,0); }

  float closest = DBL_MAX;
  int closestId = -1;
  for (int i = 0; i < spheres.size(); ++i)  {
    float dist;
    if (spheres[i].intersect(rayOrigin, rayDir, dist) && dist < closest) {
      closest = dist;
      closestId = i;
    }
  }

  static const Vec3f skyColor(0.7, 0.7, 1.0);
  if (closestId < 0) { return skyColor; } // no hit.. sky color

  const Sphere& hitSphere = spheres[closestId];

  const Vec3f intersectPoint = rayOrigin + rayDir * closest;
  const Vec3f normal = (intersectPoint - hitSphere.center).norm();
  const Vec3f surfaceColor = hitSphere.color;

  const Vec3f directLightColor = surfaceColor.cwiseMult(directLight(
    intersectPoint, normal, spheres, lightIndices)) *
    (1.0f - hitSphere.specularCoef);

  Vec3f indirectLightColor;
  if (randf01() < hitSphere.specularCoef) { // specular/mirror
    const Vec3f reflectDir = rayDir - normal * (2.0f * normal.dot(rayDir));
    const Vec3f indirectDir =
      (reflectDir + randomUnitVec() * hitSphere.specularRadius).norm();
    indirectLightColor = pathTrace(
      intersectPoint, indirectDir, spheres, lightIndices, depth + 1);
  } else { // diffuse
    const Vec3f indirectDir = randomUnitVecCosDistrib(normal);
    indirectLightColor = surfaceColor.cwiseMult(pathTrace(
      intersectPoint, indirectDir, spheres, lightIndices, depth + 1));
  }
  return hitSphere.emitColor + directLightColor + indirectLightColor;
}

static uint8_t* mapPixelsThreadPool(
  const int w,
  const int h,
  const function<Vec3f(const int, const int)>& f
) {
  static const int kNumThreads = 12;
  static const int kNumChannels = 3;
  static const float kGamma = 1.0 / 2.2;
  uint8_t* imageBytes = new uint8_t[w * h * kNumChannels];

  int currY = 0;
  std::mutex currYMutex; // protects currY
  vector<std::thread> threads;
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&] {
      while(true) {
        int y;
        {
          std::lock_guard<std::mutex> lock(currYMutex);
          if (currY >= h) { return; }
          y = currY;
          ++currY;
        } // unlock currYMutex and do work

        for (int x = 0; x < w; ++x) {
          const Vec3f color = f(x, y);
          imageBytes[(y * w + x) * kNumChannels + 0] = std::min(1.0f, powf(color.x, kGamma)) * 255.0f;
          imageBytes[(y * w + x) * kNumChannels + 1] = std::min(1.0f, powf(color.y, kGamma)) * 255.0f;
          imageBytes[(y * w + x) * kNumChannels + 2] = std::min(1.0f, powf(color.z, kGamma)) * 255.0f;
        }
      } // end while(true)
    }); // end lambda
  }
  for (auto& t : threads) { t.join(); }

  return imageBytes;
}

static void mapXYThreadPool(
  const int w,
  const int h,
  const function<void(const int, const int)>& f
) {
  static const int kNumThreads = 12;

  int currY = 0;
  std::mutex currYMutex; // protects currY
  vector<std::thread> threads;
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&] {
      while(true) {
        int y;
        {
          std::lock_guard<std::mutex> lock(currYMutex);
          if (currY >= h) { return; }
          y = currY;
          ++currY;
        } // unlock currYMutex and do work

        for (int x = 0; x < w; ++x) {
          f(x, y);
        }
      } // end while(true)
    }); // end lambda
  }
  for (auto& t : threads) { t.join(); }
}


Vec3f primaryRayColor(
  vector<Sphere>& spheres,
  vector<int>& lightIndices,
  const int numSamples,
  const int w,
  const int h,
  const int x,
  const int y
) {
  const Vec3f rayOrigin(0, 25, -49);
  Vec3f color(0, 0, 0);
  for (int sample = 0; sample < numSamples; ++sample) {
    const Vec3f rayDir = Vec3f(
      x - w/2 + randf01() - 0.5f,
      y - h/2 + randf01() - 0.5f,
      1.0*h).norm();

    color = color + pathTrace(rayOrigin, rayDir, spheres, lightIndices);
  }
  color = color / float(numSamples);
  return color;
}

static inline uint8_t getPixel(
  const uint8_t* imageBytes,
  const int w,
  const int h,
  const int numChannels,
  int x,
  int y,
  const int c
) {
  x = std::max(0, std::min(w - 1, x));
  y = std::max(0, std::min(h - 1, y));
  return imageBytes[(y * w + x) * numChannels + c];
}

vector<float> getPatchFeature(
  const uint8_t* imageBytes,
  const int w,
  const int h,
  const int numChannels,
  const int patchRadius,
  const int x,
  const int y
) {
  vector<float> patchFv;
  patchFv.push_back(float(x) / float(w));
  patchFv.push_back(float(y) / float(h));
  for (int j = -patchRadius; j <= patchRadius; ++j) {
    for (int i = -patchRadius; i <= patchRadius; ++i) {
      const float r = getPixel(imageBytes, w, h, numChannels, x + i, y + j, 0) / 255.0f;
      const float g = getPixel(imageBytes, w, h, numChannels, x + i, y + j, 1) / 255.0f;
      const float b = getPixel(imageBytes, w, h, numChannels, x + i, y + j, 2) / 255.0f;
      patchFv.push_back(r);
      patchFv.push_back(g);
      patchFv.push_back(b);
    }
  }
  return patchFv;
}

int main(int argc, char** argv) {

  vector<Sphere> spheres = {
    //            center        rad   surface color      emit
    Sphere(Vec3f(-10000, 0, 0), 9950, Vec3f(1, 0, 0),             Vec3f(0, 0, 0), 0, 0),
    Sphere(Vec3f(10000, 0, 0),  9950, Vec3f(0, 1, 0),             Vec3f(0, 0, 0), 0, 0),
    Sphere(Vec3f(0, 10000, 0),  9950, Vec3f(1, 1, 1),             Vec3f(0, 0, 0), 0, 0),
    Sphere(Vec3f(0, -10000, 0), 9950, Vec3f(0, 0, 1),             Vec3f(0, 0, 0), 0, 0),
    Sphere(Vec3f(0, 0, -10000), 9950, Vec3f(0, 0.5, 1),           Vec3f(0, 0, 0), 0, 0),
    Sphere(Vec3f(0, 0, 10000),  9950, Vec3f(1, 1, 1),             Vec3f(0, 0, 0), 0, 0),

    Sphere(Vec3f(30, 38,  20),  12,    Vec3f(1, 1, 1),            Vec3f(0, 0, 0), 0, 0),
    Sphere(Vec3f(-30, 38, 20),  12,    Vec3f(1, 1, 1),            Vec3f(0, 0, 0), 0.5, 0.2),
    Sphere(Vec3f(0, 38,   20),  12,    Vec3f(1, 1, 1),            Vec3f(0, 0, 0), 1, 0),

    Sphere(Vec3f(20, -10, 0),    5,   Vec3f(1, 1, 1),             Vec3f(0.5, 0.5, 0.5), 0, 0),
  };

  vector<int> lightIndices; // indexes into spheres of all that are emissive
  for (int i = 0; i < spheres.size(); ++i) {
    if (spheres[i].emitColor.dot(spheres[i].emitColor) > 0) {
      lightIndices.push_back(i);
    }
  }

  cout << "rendering first pass" << endl;
  auto startTime = Clock::now();
  static const int w = 1920/2;
  static const int h = 1080/2;
  static const int kNumSamples = 100;
  static const int kNumChannels = 3;
  static const int kNumSamplesHi = 2000;
  static const float kHiSampleFrac = 0.01;
  static const int kPatchRadius = 2;

  uint8_t* imageBytes = mapPixelsThreadPool(w, h, [&](const int x, const int y) {
    return primaryRayColor(spheres, lightIndices, kNumSamples, w, h, x, y);
  });
  auto endTime = Clock::now();
  cout << "render time: " << 1.0E-9 * chrono::duration_cast<chrono::nanoseconds>(endTime - startTime).count() << " sec" << endl;
  stbi_write_png("out.png", w, h, kNumChannels, imageBytes, w * kNumChannels);

  cout << "gathering examples for de-noising" << endl;
  auto startDenoiseSampleTime = Clock::now();
  std::mutex trainingDataMutex;
  vector<vector<float>> features;
  vector<float> targetsR;
  vector<float> targetsG;
  vector<float> targetsB;
  vector<float> instWeights;
  mapXYThreadPool(w, h, [&](const int x, const int y) {
    if (randf01() < kHiSampleFrac) {
      const Vec3f hiQualityColor = primaryRayColor(
        spheres, lightIndices, kNumSamplesHi, w, h, x, y);
      const vector<float> patchFV = getPatchFeature(
        imageBytes, w, h, kNumChannels, kPatchRadius, x, y);

      std::lock_guard<std::mutex> lock(trainingDataMutex);
      features.push_back(patchFV);
      targetsR.push_back(hiQualityColor.x);
      targetsG.push_back(hiQualityColor.y);
      targetsB.push_back(hiQualityColor.z);
      instWeights.push_back(1.0f);
    }
  });
  auto endDenoiseSampleTime = Clock::now();
  cout << "denoise sample time: " << 1.0E-9 * chrono::duration_cast<chrono::nanoseconds>(endDenoiseSampleTime - startDenoiseSampleTime).count() << " sec" << endl;

  cout << "training de-noising model" << endl;
  cout << "# examples=" << instWeights.size() << endl;
  auto startTrainTime = Clock::now();
  static const int kNumTrees = 14;
  static const int kTreeHeight = 18;
  const auto modelR = BasicEnsemble::train(kNumTrees, kTreeHeight, features, targetsR, instWeights);
  const auto modelG = BasicEnsemble::train(kNumTrees, kTreeHeight, features, targetsG, instWeights);
  const auto modelB = BasicEnsemble::train(kNumTrees, kTreeHeight, features, targetsB, instWeights);
  auto endTrainTime = Clock::now();
  cout << "train time: " << 1.0E-9 * chrono::duration_cast<chrono::nanoseconds>(endTrainTime - startTrainTime).count() << " sec" << endl;

  cout << "denoising with model" << endl;
  auto startDenoiseTime = Clock::now();
  uint8_t* imageDenoised = mapPixelsThreadPool(w, h, [&](const int x, const int y) {
    const vector<float> v = getPatchFeature(
      imageBytes, w, h, kNumChannels, kPatchRadius, x, y);
    return Vec3f(modelR.f(v), modelG.f(v), modelB.f(v));
  });
  auto endDenoiseTime = Clock::now();
  cout << "denoise time: " << 1.0E-9 * chrono::duration_cast<chrono::nanoseconds>(endDenoiseTime - startDenoiseTime).count() << " sec" << endl;

  stbi_write_png("out_z.png", w, h, kNumChannels, imageDenoised, w * kNumChannels);

  delete[] imageBytes;
  delete[] imageDenoised;
  return EXIT_SUCCESS;
}
