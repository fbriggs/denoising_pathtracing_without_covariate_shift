/*
Copyright 2017 Forrest Briggs

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

I am providing code in this repository to you under an open source license. Because this is my personal repository, the license you receive to my code is from me, and not from my employer (Facebook).
*/

#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <limits>

namespace hyper_decision_tree {

using namespace std;

struct HeapRegressionTree {

  static const int kRoot = 0;
  static const int kLeaf = -1;

  int height, numNodes, numLeaves, numInterior;
  vector<float> w;  // "weights" - for interior nodes this is a threshold. for leaves, it is the regression output.
  vector<int> idx;  // for interior nodes, this is a feature index. for leaves, it is kIsLeaf

  HeapRegressionTree() {}

  HeapRegressionTree(const int height) :
    height(height),
    numNodes((2 << (height - 1)) - 1),
    numLeaves(height <= 1 ? 1 : 2 << (height - 2)),
    numInterior(numNodes - numLeaves),
    w(numNodes, 0.0f),
    idx(numNodes)
  {}

  void makeLeaf(
      const int currNode,
      const vector<int>& indices,
      const vector<float>& targets,
      const vector<float>& instWeights) {

    idx[currNode] = kLeaf;

    // handle degenerate case
    if (indices.size() == 0) {
      cout << "degenerate!" << endl;
      exit(1);
    }

    float sum = 0.0;
    float sumWeight = 0.0;
    for (const int i : indices) {
      sum += targets[i] * instWeights[i];
      sumWeight += instWeights[i];
    }
    w[currNode] = sum / sumWeight;
  }

  bool allSameTarget(
      const vector<int>& indices,
      const vector<float>& targets) {

    const float t0 = targets[indices[0]];
    for (int i = 1; i < indices.size(); ++i) {
      if (targets[indices[i]] != t0) { return false; }
    }
    return true;
  }

  void optimize(
      const int currNode,
      const int featureDim,
      const vector<int>& indices,
      const vector<vector<float>>& features,
      const vector<float>& targets,
      const vector<float>& instWeights) {

    // leaf- set 'weight' to weighted average of targets
    if (isLeafLevel(currNode) || indices.size() < 2 || allSameTarget(indices, targets)) {
      makeLeaf(currNode, indices, targets, instWeights);
    // interior node. pick split index and threshold, recurse
    } else {
      // try each possible feature to split on
      static const int kNoSplitFound = -1;
      float minSplitCriteria = std::numeric_limits<float>::max();
      int bestSplit = kNoSplitFound;
      float bestThreshold = 0.0f;
      for (int split = 0; split < featureDim; ++split) {
        static const int randomThresholds = std::min(5, int(indices.size()));
        for (int guessNum = 0; guessNum < randomThresholds; ++guessNum) {
          const int randIdx1 = indices[rand() % indices.size()];
          const int randIdx2 = indices[rand() % indices.size()];
          const float threshold =
            (features[randIdx1][split] + features[randIdx2][split]) / 2.0f;

          // compute weighted average target for instances left/right of split
          float sumTargetLeft = 0.0f;
          float sumWeightLeft = 0.0f;
          float sumTargetRight = 0.0f;
          float sumWeightRight = 0.0f;
          for (const int i : indices) {
            if (features[i][split] < threshold) {
              sumTargetLeft += targets[i] * instWeights[i];
              sumWeightLeft += instWeights[i];
            } else {
              sumTargetRight += targets[i] * instWeights[i];
              sumWeightRight += instWeights[i];
            }
          }

          // degenerate splits are not candidates for best split
          if (sumWeightLeft == 0.0f || sumWeightRight == 0.0f) {
            continue;
          }

          const float avgTargetLeft = sumTargetLeft / sumWeightLeft;
          const float avgTargetRight = sumTargetRight / sumWeightRight;

          // compute weighted L1-variance of targets on left/right of split
          float varLeft = 0.0f;
          float varRight = 0.0f;
          for (const int i : indices) {
            if (features[i][split] < threshold) {
              varLeft += fabs(targets[i] - avgTargetLeft) * instWeights[i];
            } else {
              varRight += fabs(targets[i] - avgTargetRight) * instWeights[i];
            }
          }
          const float splitCriteria = varLeft + varRight;
          if (splitCriteria < minSplitCriteria) {
            minSplitCriteria = splitCriteria;
            bestSplit = split;
            bestThreshold = threshold;
          }
        }

      } // end loop over candidate splits

      // if all we got is degenerate splits, make a leaf
      if (bestSplit == kNoSplitFound) {
        makeLeaf(currNode, indices, targets, instWeights);
      }

      // save the best split/threshold we found
      idx[currNode] = bestSplit;
      w[currNode] = bestThreshold;

      // partition the training examples that made it to this node to the left
      // and right children, then recurse.
      vector<int> indicesLeft;
      vector<int> indicesRight;
      for (const int i : indices) {
        if (features[i][bestSplit] < bestThreshold) {
          indicesLeft.push_back(i);
        } else {
          indicesRight.push_back(i);
        }
      }

      // if we failed to get a meaningful partition, make a leaf
      if (indicesLeft.size() == 0 || indicesRight.size() == 0) {
          makeLeaf(currNode, indices, targets, instWeights);
      } else {
        optimize(left(currNode),  featureDim, indicesLeft,  features, targets, instWeights);
        optimize(right(currNode), featureDim, indicesRight, features, targets, instWeights);
      }
    } // end interior node
  }

  static HeapRegressionTree train(
      const int height,
      const vector<vector<float>>& features,
      const vector<float>& targets,
      const vector<float>& instWeights) {

    HeapRegressionTree tree(height);
    vector<int> indices(targets.size());
    for (int i = 0; i < targets.size(); ++i) { indices[i] = i; }
    const int featureDim = features[0].size();
    tree.optimize(kRoot, featureDim, indices, features, targets, instWeights);
    return tree;
  }

  bool isLeafLevel(const int node) const { return node >= numInterior; }
  static int left(const int i)    { return 2 * i + 1; }
  static int right(const int i )  { return 2 * i + 2; }

  float f(const vector<float>& x) const {
    int i = 0;
    while(idx[i] != kLeaf) { i = x[idx[i]] < w[i] ? left(i) : right(i); }
    return w[i];
  }
};

struct BasicEnsemble {
  vector<HeapRegressionTree> trees;

  static BasicEnsemble train(
      const int numTrees,
      const int height,
      const vector<vector<float>>& features,
      const vector<float>& targets,
      const vector<float>& instWeights) {

    BasicEnsemble e;
    e.trees.resize(numTrees);
    vector<std::thread> threads;
    for (int i = 0; i < numTrees; ++i) {
      threads.emplace_back([&, i] {
        e.trees[i] = HeapRegressionTree::train(height, features, targets, instWeights);
      });
    }
    for (auto& t : threads) { t.join(); }
    return e;
  }

  float f(const vector<float>& x) const {
    float sum = 0.0f;
    for (const auto& t: trees) { sum += t.f(x); }
    return sum / float(trees.size());
  }
};

} // end namespace hyper_decision_tree
