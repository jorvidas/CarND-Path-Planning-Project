#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <fstream>
#include <algorithm>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// SIMULATOR CONSTANTS
constexpr double TIME_STEP_SIZE = 0.02;     // Time between steps in s
constexpr double MAX_S = 6945.554;
constexpr int TOTAL_LANES = 3;

// Tunable parameters
constexpr int N = 50;                       // positions to predict
constexpr double target_v = 49.0 * .44704;  // mph to m/s
constexpr double buffer_distance = 20;      // distance to keep from car ahead
constexpr double max_a = 7 * .44704;        // gas pedal down
constexpr double min_a = -7 * .44704;       // braking

// Car variables
int lane = 1;           // starts in middle lane
int current_lane = 1;   // lane is currently in
int target_lane = 1;    // lane car would ideally be in
int next_lane = 1;      // next lane the car would go to get to ideal lane
double ref_v = 0;

// Normalizes S for track wrap around takes in s value and returns the
// normalized s between 0 and MAX_S
double normalize_s(double s) {
  s += MAX_S;
  double norm_s = fmod(s, MAX_S);
  return norm_s;
}

// Shifts s values before normalizing so when looking for cars in a range, cars
// that cross over that range dont get lost
// Example: max_s is 100, car is at 90 and checking if there is someone within
// 20 units ahead, anything >100 would be normalized otherwise
double normalize_s_to_range(double s, double bottom_of_range) {
  s -= bottom_of_range;
  s = normalize_s(s);
  s += bottom_of_range;
  return s;
}

// Checks whether a point is within a certain distance of another point
// along the track's curvature
bool check_buffer(double s, double other_s,
                  double front_buffer, double back_buffer) {
  vector<double> for_min = {s, other_s};
  double bottom_of_range = *min_element(for_min.begin(), for_min.end());
  s = normalize_s_to_range(s, bottom_of_range);
  other_s = normalize_s_to_range(other_s, bottom_of_range);
  if (other_s > s - back_buffer && other_s < s + front_buffer) {return true;}
  return false;
}

// Looks at all the lanes in a range ahead of the car to slightly behind and
// returns a vector containing the speed that lane is traveling
vector<double> get_lane_speeds(map<int, vector<vector<double> > > predictions,
                               double car_s,
                               double time_step) {
  constexpr double distance_behind = 12;
  constexpr double distance_ahead = 45;

  // initialize lanes to a unrealistically high number
  vector<double> lane_speeds(TOTAL_LANES, 999);
  for (map<int, vector<vector<double> > >::iterator it = predictions.begin();
       it != predictions.end(); it++) {
    double s = it->second[time_step][4];
    int lane = it->second[time_step][5]/4;  // truncates to lane
    // if car is in one of the lanes and within the area we are looking at
    if (lane >= 0 &&
        lane < TOTAL_LANES &&
        check_buffer(car_s, s, distance_ahead, distance_behind)) {
      double vx = it->second[time_step][2];
      double vy = it->second[time_step][3];
      double v = sqrt(vx*vx+vy*vy);
      // if velocity is slower than current lane speed
      if (v < lane_speeds[lane]) {lane_speeds[lane] = v;}
    }
  }
  return lane_speeds;
}

// Returns the lane that is currently moving the fastest. Will give preference
// to the right lane in case of a tie. Includes a small bias factor for the
// current lane.
int get_target_lane(map<int, vector<vector<double> > > predictions,
                    double car_s,
                    int current_lane,
                    int time_step) {
  constexpr double current_lane_bias = 0.5;
  vector<double> lane_speeds = get_lane_speeds(predictions, car_s, time_step);
  // add bias towards staying in current lane
  lane_speeds[current_lane] = lane_speeds[current_lane] + current_lane_bias;
  int best_lane = TOTAL_LANES - 1 -
                  distance(lane_speeds.rbegin(),
                           max_element(lane_speeds.rbegin(),
                                       lane_speeds.rend()));
  return best_lane;
}

// returns the next lane for the car to move into on the way to the target lane
int get_next_lane(int current_lane, int target_lane) {
  int next_lane = current_lane;
  if (current_lane < target_lane) {
    next_lane++;
  } else if (current_lane > target_lane) {
    next_lane--;
  }
  return next_lane;
}

// checks a lane to see if the lane is clear to move into
bool check_clear(map<int, vector<vector<double> > > predictions,
                 double car_s,
                 int next_lane,
                 int time_step) {
  double lane_change_buffer = 18;
  // check all the cars
  for (map<int, vector<vector<double> > >::iterator it = predictions.begin();
       it != predictions.end();
       it++) {
    // if they are in the lane that I am going to
    if (int(it->second[time_step][5]/4) == next_lane) {
      // check if they are within my buffer range
      double other_car_s = it->second[time_step][4];
      if (check_buffer(car_s, other_car_s, lane_change_buffer,
                       lane_change_buffer)) {
        return false;
      }
    }
  }
  return true;
}

// Shifts and rotates between the world coordinates and the vehicle coordinates
vector<double> transform_to_car(double car_x, double car_y, double yaw,
                                double world_x, double world_y) {
  // Shift
  double shift_x = world_x - car_x;
  double shift_y = world_y - car_y;

  // Rotate
  double trans_x = shift_x * cos(-yaw) - shift_y * sin(-yaw);
  double trans_y = shift_x * sin(-yaw) + shift_y * cos(-yaw);
  vector<double> result = {trans_x, trans_y};
  return result;
}

// Shifts and rotates between the world coordinates and the vehicle coordinates
vector<double> transform_to_world(double car_x, double car_y, double yaw,
                                  double ref_x, double ref_y) {
  // Rotate
  double rotated_x = ref_x * cos(yaw) - ref_y * sin(yaw);
  double rotated_y = ref_x * sin(yaw) + ref_y * cos(yaw);

  // Shift
  double trans_x = rotated_x + car_x;
  double trans_y = rotated_y + car_y;

  vector<double> result = {trans_x, trans_y};
  return result;
}

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2) {
    return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, vector<double> maps_x,
                    vector<double> maps_y) {
    double closestLen = 100000;  // large number
    int closestWaypoint = 0;

    for (int i = 0; i < maps_x.size(); i++) {
        double map_x = maps_x[i];
        double map_y = maps_y[i];
        double dist = distance(x, y, map_x, map_y);
        if (dist < closestLen) {
            closestLen = dist;
            closestWaypoint = i;
        }
    }
    return closestWaypoint;
}

int NextWaypoint(double x, double y, double theta, vector<double> maps_x,
                 vector<double> maps_y) {
    int closestWaypoint = ClosestWaypoint(x, y, maps_x, maps_y);

    double map_x = maps_x[closestWaypoint];
    double map_y = maps_y[closestWaypoint];

    double heading = atan2((map_y-y), (map_x-x));

    double angle = abs(theta-heading);

    if (angle > pi()/4) {
        closestWaypoint++;
    }

    return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta,
                         vector<double> maps_x, vector<double> maps_y) {
    int next_wp = NextWaypoint(x, y, theta, maps_x, maps_y);

    int prev_wp;
    prev_wp = next_wp-1;
    if (next_wp == 0) {
        prev_wp  = maps_x.size()-1;
    }

    double n_x = maps_x[next_wp]-maps_x[prev_wp];
    double n_y = maps_y[next_wp]-maps_y[prev_wp];
    double x_x = x - maps_x[prev_wp];
    double x_y = y - maps_y[prev_wp];

    // find the projection of x onto n
    double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
    double proj_x = proj_norm*n_x;
    double proj_y = proj_norm*n_y;

    double frenet_d = distance(x_x, x_y, proj_x, proj_y);

    // see if d value is positive or negative by comparing it to a center point

    double center_x = 1000-maps_x[prev_wp];
    double center_y = 2000-maps_y[prev_wp];
    double centerToPos = distance(center_x, center_y, x_x, x_y);
    double centerToRef = distance(center_x, center_y, proj_x, proj_y);

    if (centerToPos <= centerToRef) {
        frenet_d *= -1;
    }

    // calculate s value
    double frenet_s = 0;
    for (int i = 0; i < prev_wp; i++) {
        frenet_s += distance(maps_x[i], maps_y[i], maps_x[i+1], maps_y[i+1]);
    }

    frenet_s += distance(0, 0, proj_x, proj_y);

    return {frenet_s, frenet_d};
}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, vector<double> maps_s,
                     vector<double> maps_x, vector<double> maps_y) {
    int prev_wp = -1;

    while (s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) )) {
        prev_wp++;
    }

    int wp2 = (prev_wp+1)%maps_x.size();

    double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),
                           (maps_x[wp2]-maps_x[prev_wp]));
    // the x,y,s along the segment
    double seg_s = (s-maps_s[prev_wp]);

    double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
    double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

    double perp_heading = heading-pi()/2;

    double x = seg_x + d*cos(perp_heading);
    double y = seg_y + d*sin(perp_heading);

    return {x, y};
}

// Predict where the cars will be for each of N steps into the future
// Inputs: sensor fusion data vec<vec<double> > with the inner vector holding
//         [id, x, y, vx, vy, s, d] and the outer vector holding multiple cars
// Outputs: Map with id as key and vec<vec<double> > as pair. Inner vector
//          is [x, y, vx, vy, s, d]. Outer vector contains each of the time
//          steps starting with current.
// Assumes: Cars are traveling along the center of the road and not changing
//          lanes.

map<int, vector<vector<double> > > make_predictions(vector<vector<double> > sf,
                                                      vector<double> maps_s,
                                                      vector<double> maps_x,
                                                      vector<double> maps_y) {
    // Containers
    vector<vector<double> > predictions;
    map<int, vector<vector<double> > > all_predictions;
    vector<double> prediction;

    // predict for each car
    for (auto a : sf) {
        // for map key
        int id = a[0];

        // does not change
        double d = a[6];
        double vx = a[3];
        double vy = a[4];

        // for prediction
        double s = a[5];
        double v = sqrt(vx*vx + vy*vy);

        // predict for each future time step
        for (int i = 0; i < N; i++) {
            // make predictions
            // update s for next step
            s += v * TIME_STEP_SIZE;
            s = normalize_s(s);

            vector<double> euclid_coord = getXY(s, d, maps_s, maps_x, maps_y);
            double x = euclid_coord[0];
            double y = euclid_coord[1];

            // add
            double array[6] = {x, y, vx, vy, s, d};
            prediction.insert(prediction.end(), array,
                              array+(sizeof(array)/sizeof(array[0])));

            // add time step
            predictions.push_back(prediction);
            prediction.clear();
        }
        // add prediction data to map
        all_predictions[id] = predictions;
        predictions.clear();
    }

    vector<vector<double> > a = all_predictions[0];
    vector<vector<double> > b = all_predictions[1];

    // return map
    return all_predictions;
}



int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  h.onMessage([&map_waypoints_x, &map_waypoints_y, &map_waypoints_s,
               &map_waypoints_dx,
               &map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws,
               char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    // auto sdata = string(data).substr(0, length);
    // cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object

            // Main car's localization Data
            double car_x = j[1]["x"];
            double car_y = j[1]["y"];
            double car_s = j[1]["s"];
            double car_d = j[1]["d"];
            double car_yaw = j[1]["yaw"];
            double car_speed = j[1]["speed"];
            // Previous path data given to the Planner
            auto previous_path_x = j[1]["previous_path_x"];
            auto previous_path_y = j[1]["previous_path_y"];
            // Previous path's end s and d values
            double end_path_s = j[1]["end_path_s"];
            double end_path_d = j[1]["end_path_d"];

            // Sensor Fusion Data, a list of all other cars on the same side
            // of the road.
            auto sensor_fusion = j[1]["sensor_fusion"];

            json msgJson;

            vector<double> next_x_vals;
            vector<double> next_y_vals;

            // predict
            map<int, vector<vector<double> > > predictions = make_predictions(sensor_fusion,
                                      map_waypoints_s, map_waypoints_x, map_waypoints_y);

            /*
            * SPLINE
            */

            // Spline
            tk::spline s;

            // Move car forward to get data for behavior
            // move down middle lane

            double pos_x;
            double pos_y;
            double pos_s;
            double prev_x;
            double prev_y;
            double angle;
            int path_size = previous_path_x.size();

            // Set spline
            int spline_points = 5;
            vector<double> spline_xs;
            vector<double> spline_ys;

            // If no previous path use current position
            // Otherwise use data from the end of the previous path
            if (path_size == 0) {
                pos_x = car_x;
                pos_y = car_y;
                pos_s = car_s;
                end_path_s = car_s;
                angle = deg2rad(car_yaw);
                prev_x = car_x - cos(car_yaw);
                prev_y = car_y - sin(car_yaw);
            } else {
                pos_x = previous_path_x[path_size-1];
                pos_y = previous_path_y[path_size-1];
                pos_s = end_path_s;

                prev_x = previous_path_x[path_size-2];
                prev_y = previous_path_y[path_size-2];

                angle = atan2(pos_y-prev_y, pos_x-prev_x);
            }

            // add two current points to spline
            spline_xs.push_back(prev_x);
            spline_xs.push_back(pos_x);

            spline_ys.push_back(prev_y);
            spline_ys.push_back(pos_y);

            // add future points to the spline.
            int additional_spline_anchors = 3;
            double spline_increment = 30;
            vector<double> next_spline;

            for (int i = 1; i <= additional_spline_anchors; i++) {
              next_spline = getXY(end_path_s + i * spline_increment,
                                  2 + lane * 4,
                                  map_waypoints_s, map_waypoints_x,
                                  map_waypoints_y);
              spline_xs.push_back(next_spline[0]);
              spline_ys.push_back(next_spline[1]);
            }

            // Rotate all points
            for (int i = 0; i < spline_xs.size(); i++) {
              vector<double> transformed;
              transformed = transform_to_car(pos_x, pos_y, angle,
                                             spline_xs[i], spline_ys[i]);
              spline_xs[i] = transformed[0];
              spline_ys[i] = transformed[1];
            }

            // Set spline
            s.set_points(spline_xs, spline_ys);
            spline_xs.clear();
            spline_ys.clear();

            /*
            * CREATE TRAJECTORY
            */

            // Add portion of previous path remaining
            for (int i = 0; i < path_size; i++) {
                next_x_vals.push_back(previous_path_x[i]);
                next_y_vals.push_back(previous_path_y[i]);
            }

            // Add new points on to complete path
            int steps_to_add = N-path_size;

            // Get target lane
            current_lane = end_path_d / 4;
            target_lane = get_target_lane(predictions, end_path_s,
                                          current_lane, path_size - 1);
            next_lane = get_next_lane(current_lane, target_lane);

            for (int i = 1; i <= steps_to_add; i++) {
              // Get a
              bool too_close = false;
              double ref_a = 0;
              int additional_change_lane_distance = 10;
              int current_time_step = i + path_size - 1;

              pos_s += ref_v*TIME_STEP_SIZE;
              pos_s = normalize_s(pos_s);

              // loop over cars
              for (map<int, vector<vector<double> > >::iterator it = predictions.begin();
                   it != predictions.end();
                   it++) {
                // check to see if they are in our lane
                // check before we slow down to see if it makes sense to change
                // lanes
                  if (it->second[current_time_step][5] >= 4*lane &&
                      it->second[current_time_step][5] <= 4*(lane + 1)) {
                  double other_car_s = it->second[current_time_step][4];
                  // check s for lane change, account for s wrap
                  if (check_buffer(pos_s, other_car_s, buffer_distance +
                                   additional_change_lane_distance, 0)) {
                  // check for speed sync
                    if (check_buffer(pos_s, other_car_s, buffer_distance, 0)) {
                      int lead_car_vx = it->second[current_time_step][2];
                      int lead_car_vy = it->second[current_time_step][3];
                      double lead_car_speed = sqrt(lead_car_vx*lead_car_vx +
                                                   lead_car_vy*lead_car_vy);
                      // check to see if we are not already slower than them
                      if (lead_car_speed < car_speed + 3) {
                        too_close = true;
                      }
                    }
                    if (check_clear(predictions, pos_s, next_lane,
                                    current_time_step)) {lane = next_lane;}
                  }
                }
              }

              if (too_close) {
                ref_a = min_a;
              } else {
                ref_a = max_a;
              }
              ref_v += ref_a * TIME_STEP_SIZE;
              // Check velocity isn't above target
              ref_v = min(ref_v, target_v);

              // get x increment for distance equal to what is
              // expected of our speed
              double d_increment = ref_v*TIME_STEP_SIZE;
              double total_distance = steps_to_add * d_increment;
              double approx_x = 30;  // made up to give it enough time to curve
              double approx_y = s(total_distance);
              double approx_angle = atan2(approx_y, approx_x);
              double x_increment = cos(approx_angle) * d_increment;

              // Get x value
              double next_x = i * x_increment;

              // Get y value
              double next_y = s(next_x);
              // Rotate
              vector<double> next_position = transform_to_world(pos_x,
                                                                pos_y,
                                                                angle,
                                                                next_x,
                                                                next_y);
              // Add to next values
              next_x_vals.push_back(next_position[0]);
              next_y_vals.push_back(next_position[1]);
            }

            // add new path for message to simulator
            msgJson["next_x"] = next_x_vals;
            msgJson["next_y"] = next_y_vals;

            auto msg = "42[\"control\","+ msgJson.dump()+"]";

            // this_thread::sleep_for(chrono::milliseconds(100));
            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      }
    } else {
      // Manual driving
      std::string msg = "42[\"manual\",{}]";
      ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}

