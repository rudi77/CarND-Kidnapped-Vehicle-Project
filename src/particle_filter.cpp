/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <sstream>
#include <string>
#include <iterator>
#include <iostream>
#include <limits>

#include "particle_filter.h"

using namespace std;

static void add_noise(Particle& particle, default_random_engine& gen, double std[]);

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  auto initial_weight = 1.0 / num_particles;

  // The num_particles is set in the constructor either with the default
  // value of 500 particles or by providing the number of particles as a 
  // ctor argument.

    // Create weights with default values
  weights.assign(num_particles, initial_weight);

  default_random_engine gen;

  // Generate N particles and insert them into the particles vector
  for (auto i = 0; i < num_particles; i++)
  {
    Particle particle;

    particle.id = i;
    particle.x = x;
    particle.y = y;
    particle.theta = theta;
    particle.weight = initial_weight;

    // add some noise
    add_noise(particle, gen, std);

    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
  default_random_engine gen;

  auto v_over_yaw_rate = velocity / yaw_rate;
  auto yaw_rate_times_delta_t = yaw_rate * delta_t;
  auto v_times_delta_t = velocity * delta_t;

  // Iterate over each particle and update x, y and theta. If yaw_rate is zero or in this case below
  // a certain threshold then slightly different formulas are used for state computation.
  for (auto& p : particles) 
  {
    if (fabs(yaw_rate) > 0.001) 
    {
      p.x += v_over_yaw_rate * (sin(p.theta + yaw_rate_times_delta_t) - sin(p.theta));
      p.y += v_over_yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate_times_delta_t));
      p.theta += yaw_rate_times_delta_t;
    }
    else 
    {
      p.x += v_times_delta_t * cos(p.theta);
      p.y += v_times_delta_t * sin(p.theta);
    }

    add_noise(p, gen, std_pos);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations)
{
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	// observed measurement to this particular landmark.

  auto associated_landmarks = vector<LandmarkObs>();

  LandmarkObs associated_landmark;

  // ForEach observed landmark:
  //  Iterate over each predicted landmark and calculate the distance between observed and predicted landmark
  //  Predicted landmark that is closest to observed landmark wins
  for (auto& observed_landmark : observations)
  {
    // Initial closest distance
    auto closest_dist = numeric_limits<double>::max();
    
    for (auto& predicted_landmark : predicted)
    {
      // calc distance and compare it with closest_dist
      auto distance = dist(observed_landmark.x, observed_landmark.y, predicted_landmark.x, predicted_landmark.y);
      
      if (distance < closest_dist)
      {
        closest_dist = distance;
        associated_landmark = predicted_landmark;
      }
    }

    associated_landmarks.push_back(associated_landmark);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], vector<LandmarkObs> observations, Map map_landmarks)
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution

	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  for (auto& particle : particles)
  {
    // 1.) Get all landmarks within the sensor's range
    vector<LandmarkObs> predicted_landmarks;

    for (auto& single_landmark : map_landmarks.landmark_list)
    {
      auto distance = dist(particle.x, particle.y, single_landmark.x_f, single_landmark.y_f);

      if (distance <= sensor_range)
      {
        LandmarkObs landmark = { single_landmark.id_i, single_landmark.x_f, single_landmark.y_f };

        predicted_landmarks.push_back(landmark);
      }
    }


    // 2.) Coordinate transformation of obervations.
    vector<LandmarkObs> transformed_landmarks;

    for (auto& observed_landmark : observations)
    {
      auto transformed_coordinates = transform_coordinates(particle.x, particle.y, particle.theta, observed_landmark.x, observed_landmark.y);
      
      auto transformed_observation = LandmarkObs();
      transformed_observation.id = observed_landmark.id;
      transformed_observation.x = std::get<0>(transformed_coordinates);
      transformed_observation.y = std::get<1>(transformed_coordinates);

      transformed_landmarks.push_back(transformed_observation);
    }


    // 2.) Data association
    dataAssociation(predicted_landmarks, transformed_landmarks);

    // 3.) Update weights

  }


  //double multiplier = 1.0 / (2 * M_PI * landmark_std_x * landmark_std_y);
  //double cov_x = pow(landmark_std_x, 2.0);
  //double cov_y = pow(landmark_std_y, 2.0);

  //double observation_prob_i = exp(-pow(measurement.x - nearest_landmark.x, 2.0) / (2.0*cov_x) - pow(measurement.y - nearest_landmark.y, 2.0) / (2.0*cov_y));
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best) 
{
  auto v = best.associations;

  if (v.size() == 0) return "";

  auto func = [](string a, int b) { return a + " " + to_string(b); };

  auto s = std::accumulate(next(v.begin()), v.end(), to_string(v[0]), func);

  return s;
}

string ParticleFilter::getSenseX(Particle best)
{
  auto v = best.sense_x;

  if (v.size() == 0) return "";

  auto func = [] (string a, double b) { return a + " " + to_string(b); };

  auto s = std::accumulate(next(v.begin()), v.end(), to_string(v[0]), func);

  return s;
}

string ParticleFilter::getSenseY(Particle best)
{
  auto v = best.sense_y;

  if (v.size() == 0) return "";

  auto func = [](string a, double b) { return a + " " + to_string(b); };

  auto s = std::accumulate(next(v.begin()), v.end(), to_string(v[0]), func);

  return s;
}

static void add_noise(Particle& particle, default_random_engine& gen, double std[])
{
  normal_distribution<> gaussian_x(particle.x, std[0]);
  normal_distribution<> gaussian_y(particle.y, std[1]);
  normal_distribution<> gaussian_theta(particle.theta, std[2]);

  particle.x = gaussian_x(gen);
  particle.y = gaussian_y(gen);
  particle.theta = gaussian_theta(gen);
}
