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
#include <assert.h>

#include "particle_filter.h"

using namespace std;

static void add_noise(Particle& particle, default_random_engine& gen, double std[]);

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  auto initial_weight = 1.0 / num_particles;

  // The num_particles is set in the constructor either with the default
  // value of 200 particles or by providing the number of particles as a 
  // ctor argument.

  // Create weights with default values
  weights.assign(num_particles, initial_weight);

  default_random_engine gen;

  // Generate N particles and insert them into the particles vector
  for (auto i = 0; i < num_particles; i++)
  {
    auto particle = Particle::create(i, x, y, theta, initial_weight);

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

vector<LandmarkObs> ParticleFilter::dataAssociation(vector<LandmarkObs> landmarks, vector<LandmarkObs>& observations)
{
  auto associated_landmarks = vector<LandmarkObs>();

  LandmarkObs associated_landmark;

  // ForEach observed landmark:
  //  Iterate over each landmark and calculate the distance between observed landmark and landmark
  //  The observed landmark that is closest to landmark wins
  for (auto& observed_landmark : observations)
  {
    // Initial closest distance
    auto closest_dist = numeric_limits<double>::max();
    
    for (auto& landmark : landmarks)
    {
      auto distance = dist(observed_landmark.x, observed_landmark.y, landmark.x, landmark.y);
      
      if (distance < closest_dist)
      {
        closest_dist = distance;
        associated_landmark = landmark;
      }
    }

    associated_landmarks.push_back(associated_landmark);
  }

  return associated_landmarks;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], vector<LandmarkObs> observations, Map map_landmarks)
{
  auto weight_index = 0;

  for (auto& particle : particles)
  {
    // 1.) Get all landmarks within the particle's range. Therefore,
    //     calculate the distance between the particle and the landmark.
    //     Take landmark if the distance is smaller or equal the sensor's range.
    vector<LandmarkObs> landmarks_in_range;

    for (auto& single_landmark : map_landmarks.landmark_list)
    {
      auto distance = dist(particle.x, particle.y, single_landmark.x_f, single_landmark.y_f);

      if (distance <= sensor_range)
      {
        LandmarkObs landmark = { single_landmark.id_i, single_landmark.x_f, single_landmark.y_f };

        landmarks_in_range.push_back(landmark);
      }
    }

    // 2.) Coordinate transformation of obervations.
    vector<LandmarkObs> transformed_observations;

    for (auto& observed_landmark : observations)
    {
      auto transformed_coordinates = transform_coordinates(particle.x, particle.y, particle.theta, observed_landmark.x, observed_landmark.y);
      auto transformed_observation = LandmarkObs::Create(observed_landmark.id, transformed_coordinates.x, transformed_coordinates.y);

      transformed_observations.push_back(transformed_observation);
    }

    // 3.) Data association
    auto associated_landmark = dataAssociation(landmarks_in_range, transformed_observations);

    // 4.) Update weights
    //     Calc probability for each observation and then multiply 
    //     the probabilities with each other.
    assert(transformed_observations.size() == associated_landmark.size());

    auto weight = 1.0;

    for (auto i = 0; i < transformed_observations.size(); i++)
    {
      // define them here for better readability. compiler will optimize it.
      auto x = transformed_observations[i].x;
      auto y = transformed_observations[i].y;
      auto mu_x = associated_landmark[i].x;
      auto mu_y = associated_landmark[i].y;

      auto p = multivariate_normal_pdf(x, y, mu_x, mu_y, std_landmark[0], std_landmark[1]);

      weight *= p;

      //particle.associations.push_back(associated_landmark[i].id);
    }

    weights[weight_index++] = weight;

    particle.weight = weight;
  }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  discrete_distribution<> d(weights.begin(), weights.end());
  
  vector<Particle> resampled_particles;

  for (auto i = 0; i < num_particles; i++)
  {
    auto idx = d(gen);
    resampled_particles.push_back(particles[idx]);
  }

  assert(particles.size() == resampled_particles.size());

  particles = resampled_particles;
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
