#include <iostream>
#include <tuple>
#include <math.h>
#include <vector>
#include <limits>
#include <algorithm>


#include "../../../src/helper_functions.h"

using namespace std;

int main()
{
  // Measurement noise
  auto stddev_x = 0.3;
  auto stddev_y = 0.3;

  // landmarks
  LandmarkObs l1 = { 1, 5.0, 3.0 };
  LandmarkObs l2 = { 2, 2.0, 1.0 };
  LandmarkObs l3 = { 3, 6.0, 1.0 };
  LandmarkObs l4 = { 4, 7.0, 4.0 };
  LandmarkObs l5 = { 5, 4.0, 7.0 };

  auto landmarks = { l1, l2, l3, l4, l5 };

  // particle P
  auto p_x = 4;
  auto p_y = 5;
  auto p_theta = 0.5 * M_PI;  // 0

  ////// Coordinates of observations are provided in vehicle's coordinate system //////

  // first observation
  auto obs1_x = 2.0;
  auto obs1_y = 2.0;

  // second observation
  auto obs2_x = 3.0;
  auto obs2_y = -2.0;

  // third observation
  auto obs3_x = 0.0;
  auto obs3_y = -4.0;

  // Transform measurement coordinates
  auto trans_obs1 = transform_coordinates(p_x, p_y, p_theta, obs1_x, obs1_y);
  auto trans_obs2 = transform_coordinates(p_x, p_y, p_theta, obs2_x, obs2_y);
  auto trans_obs3 = transform_coordinates(p_x, p_y, p_theta, obs3_x, obs3_y);

  std::cout << "Transformed_OBS1: (" << trans_obs1.x << "," << trans_obs1.y << ")" << endl;
  std::cout << "Transformed_OBS2: (" << trans_obs2.x << "," << trans_obs2.y << ")" << endl;
  std::cout << "Transformed_OBS3: (" << trans_obs3.x << "," << trans_obs3.y << ")" << endl;

  // Calculate associations
  auto trans_observations = { trans_obs1, trans_obs2, trans_obs3 };
  
  vector<int> associated_landmark_id;

  int id;

  for (auto& t_obs : trans_observations)
  {
    // Initial closest distance
    auto closest_dist = numeric_limits<double>::max();

    for (auto& landmark : landmarks)
    {
      auto distance = dist(t_obs.x, t_obs.y, landmark.x, landmark.y);

      if (distance < closest_dist)
      {
        closest_dist = distance;
        id = landmark.id;
      }
    }

    associated_landmark_id.push_back(id);
  }

  auto func = [](string a, int b) { return a + " " + to_string(b); };
  auto s = std::accumulate(next(associated_landmark_id.begin()), associated_landmark_id.end(), to_string(associated_landmark_id[0]), func);

  std::cout << "Associations " << s << std::endl;

  // Calculate gaussian probabilities for each observation
  auto p_obs1 = multivariate_normal_pdf(trans_obs1.x, trans_obs1.y, l1.x, l1.y, stddev_x, stddev_y);
  auto p_obs2 = multivariate_normal_pdf(trans_obs2.x, trans_obs2.y, l2.x, l2.y, stddev_x, stddev_y);
  auto p_obs3 = multivariate_normal_pdf(trans_obs3.x, trans_obs3.y, l2.x, l2.y, stddev_x, stddev_y);

  std::cout << "P_OBS1 " << p_obs1 << " P_OBS2 " << p_obs2 << " P_OBS3 " << p_obs3 << endl;

  // Calculate final weight
  std::cout << "Total Probability: " << p_obs1 * p_obs2 * p_obs3 << endl;

  return 0;
}