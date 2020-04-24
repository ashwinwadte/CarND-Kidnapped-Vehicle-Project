/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

// random engine
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 500;  // TODO: Set the number of particles

  // create a normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(x, std[0]);
  // create a normal (Gaussian) distribution for y
  normal_distribution<double> dist_y(y, std[1]);
  // create a normal (Gaussian) distribution for theta
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i)
  {
    Particle particle;                 // create particle
    particle.id = i;
    particle.x = dist_x(gen);         // add random Gaussian noise to each particle.
    particle.y = dist_y(gen);         // add random Gaussian noise to each particle.
    particle.theta = dist_theta(gen); // add random Gaussian noise to each particle.
    particle.weight = 1.0;
    particles.push_back(particle);
  }
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  // create a normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(0, std_pos[0]);
  // create a normal (Gaussian) distribution for y
  normal_distribution<double> dist_y(0, std_pos[1]);
  // create a normal (Gaussian) distribution for theta
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; ++i)
  {
    if (fabs(yaw_rate) < 0.00001) // avoid division per small numbers / 0
    {
      // x_f = x_0 + v * dt * cos(θ), add x component of velocity in dt
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      // y_f = y_0 + v * dt * sin(θ), add y component of velocity in dt
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
      // θ_f = θ_0
      // particles[i].theta remains equal
    }
    else
    {
      // x_f = x_0 + v/θ°[sin(θ_0 ​+ θ°(dt))−sin(θ_0​)]
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      // y_f = y_0 + v/θ°[cos(θ_0) ​- cos(θ_0 + θ°(dt))]
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      // θ_f = θ_0 + θ°(dt)
      particles[i].theta += yaw_rate * delta_t;
    }

    // add random Gaussian noise to each particle.
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (unsigned int i = 0; i < observations.size(); ++i)
  {

    // first observation
    LandmarkObs observation = observations[i];

    // init minimum distance
    double min_dist = numeric_limits<double>::max();

    // init id of landmark from map placeholder to be associated with the observation
    int associated_id = -1;

    for (unsigned int j = 0; j < predicted.size(); ++j)
    {
      // first prediction
      LandmarkObs prediction = predicted[j];

      // distance between observation and prediction
      double current_distance = dist(observation.x, observation.y, prediction.x, prediction.y);

      // find the predicted landmark nearest the current observed landmark
      if (current_distance < min_dist)
      {
        min_dist = current_distance;
        associated_id = prediction.id;
      }
    }

    // set the observation's id to the nearest predicted landmark's id
    observations[i].id = associated_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  double x_map;
  double y_map;

  // loop over the particles
  for (int i = 0; i < num_particles; ++i)
  {
    // vector to be passed to dataAssociation
    vector<LandmarkObs> predicted;

    // loop over the landmarks
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j)
    {
      // take into account landmarks in sensor range
      bool sensor_range_x = false;

      if ((fabs(map_landmarks.landmark_list[j].x_f - particles[i].x) <= sensor_range))
        sensor_range_x = true;

      bool sensor_range_y = false;

      if ((fabs(map_landmarks.landmark_list[j].y_f - particles[i].y) <= sensor_range))
        sensor_range_y = true;

      if (sensor_range_x && sensor_range_y)
      {
        predicted.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
      }
    }
    // vector to be passed to dataAssociation
    vector<LandmarkObs> map_observations;
    for (unsigned int j = 0; j < observations.size(); j++)
    {
      // transform to map x coordinate double x_map;
      x_map = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
      // transform to map y coordinate
      y_map = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
      map_observations.push_back(LandmarkObs{observations[j].id, x_map, y_map});
    }

    // data association
    dataAssociation(predicted, map_observations);

    // reinitialize weight
    particles[i].weight = 1.0;

    for (unsigned j = 0; j < map_observations.size(); ++j)
    {
      // calculate normalization term
      double gauss_norm;
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];
      gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

      // calculate exponent
      double exponent;
      double mu_x = map_observations[j].x;
      double mu_y = map_observations[j].y;
      double x_obs, y_obs;

      // check that id matches
      for (unsigned k = 0; k < predicted.size(); ++k)
      {
        if ((map_observations[j].id == predicted[k].id))
        {
          x_obs = predicted[k].x;
          y_obs = predicted[k].y;
        }
      }

      exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2))) + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

      // calculate weight using normalization terms and exponent
      particles[i].weight *= gauss_norm * exp(-exponent);
    }
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> resampled_particles;

  // random index
  uniform_int_distribution<int> index_distribution(0, num_particles - 1);
  int index = index_distribution(gen);

  double beta = 0.0;

  // cache max weight
  vector<double> weights;
  for (int i = 0; i < num_particles; i++)
  {
    weights.push_back(particles[i].weight);
  }
  double max_weight = *max_element(weights.begin(), weights.end());

  // beta distribution
  uniform_real_distribution<double> beta_distribution(0.0, 1);

  for (int i = 0; i < num_particles; i++)
  {
    // update beta
    beta += beta_distribution(gen) * 2.0 * max_weight;
    while (beta > weights[index])
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }

  particles = resampled_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}