#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // set state dimension
  n_x_ = 5;

  // Set Augmented State dimension
  n_aug_ = 7;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 6;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  lambda_ = 3 - n_x_;

  // Initiale the state vector
  x_ << 0, 0, M_PI/4, 0.30, 0.18;

  // Initialize covariance matrix
  P_ << 0.3, 0, 0, 0, 0,
        0, 0.2, 0, 0, 0,
        0, 0, 0.3, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  //create augmented mean vector
  x_aug_ = VectorXd(7);
  
  //create augmented state covariance
  P_aug_ = MatrixXd(n_aug_, n_aug_);
  
  // Set Sigma Pts.
  n_sigma_pts_ = 2 * n_aug_ + 1;
  
  // Initialize Sigma prediction points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sigma_pts_);
  
  // Initialize Sigma Augumentated points matrix
  Xsig_aug_ = MatrixXd(n_aug_, n_sigma_pts_);

  // Initialize weights of prediction layer.
  weights_ = VectorXd(n_sigma_pts_);

  // Initialize measurement noise covariance matrix
  R_radar_ = MatrixXd(3,3);
  
  R_radar_ << std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0,std_radrd_*std_radrd_;
  
  R_lidar_ = Eigen::MatrixXd(2,2);
  R_lidar_ << std_laspx_*std_laspx_,0,
              0,std_laspy_*std_laspy_;

  // set weights
  double weights_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weights_0;
  for (int i=1; i < 2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      Eigen::VectorXd rawMeasurment =  meas_package.raw_measurements_;
      float rho = rawMeasurment[0];
      float phi = rawMeasurment[1];
      float rhodot = rawMeasurment[2];
      float px = rho * cos(phi);
      float py = rho * sin(phi);
      float vx = rhodot * cos(phi);
      float vy = rhodot * sin(phi);
      float vel = sqrt(vx*vx + vy * vy);
      x_ << px, py, vel, 0, 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      Eigen::VectorXd rawMeasurment =  meas_package.raw_measurements_;
      float px = rawMeasurment[0];
      float py = rawMeasurment[1];
      if (fabs(px) < 0.001) {
        px = 0.001;
      }
      if (fabs(py) < 0.001) {
        py = 0.001;
      }
      x_ << px, py, 0, 0, 0;
    }
    previous_timestamp_ =  meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }
  double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ =  meas_package.timestamp_;
  this->UKF::Prediction(delta_t);
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    this->UKF::UpdateRadar(meas_package);
  } else {
    this->UKF::UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  //Reset augmented mean state
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;

  //Reset augmented covariance matrix
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(n_x_,n_x_) = P_;
  P_aug_(5,5) = std_a_*std_a_;
  P_aug_(6,6) = std_yawdd_*std_yawdd_;


  this->UKF::generateSigmaPoints();
  //predict sigma points
  this->UKF::predictSigmaPoints(delta_t);
  
  // Predict New State and Co-variance
  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-= 2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+= 2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }

  
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Set radar measurment dimensions.
  n_z = 2;
  // Initialize radar measurment sigma points
  Zsig_ = Xsig_pred_.block(0,0,n_z, 2 * n_aug_ + 1);
  // Initialize radar measurment prediction mean
  z_pred_ = VectorXd(n_z);
  
  // Initialize measurement covariance matrix S
  S_ = MatrixXd(n_z,n_z);
  this->UKF::predictMeasurment(R_lidar_);
  this->UKF::UpdateMeasurment(meas_package);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  //transform sigma points into radar measurement space
  // Set radar measurment dimensions.
  n_z = 3;
  // Initialize radar measurment sigma points
  Zsig_ = MatrixXd(n_z, 2 * n_aug_ + 1);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig_(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig_(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig_(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }
  

  // Initialize radar measurment prediction mean
  z_pred_ = VectorXd(n_z);
  
  // Initialize measurement covariance matrix S
  S_ = MatrixXd(n_z,n_z);
  this->UKF::predictMeasurment(R_radar_);
  this->UKF::UpdateMeasurment(meas_package);
}

/**
*
* Generate sigma points on every new state and covariance.
*/
void UKF::generateSigmaPoints() {
  // Generate sigma points by normal convention.
  // //calculate square root of P
  // MatrixXd A = P_.llt().matrixL();
  // //set first column of sigma point matrix
  // Xsig_pred_.col(0)  = x_;
  // //set remaining sigma points
  // for (int i = 0; i < n_x; i++)
  // {
  //   Xsig_pred_.col(i+1)     = x_ + sqrt(lambda+n_x) * A.col(i);
  //   Xsig.col(i+1+n_x) = x - sqrt(lambda+n_x) * A.col(i);
  // }

  // std::cout << "Sigma Points: " << std::endl << Xsig_pred_ << std::endl;

  // Generate Sigma points with Augumentation
  //calculate square root of P
  MatrixXd L = P_aug_.llt().matrixL();
  //create augmented sigma points
  Xsig_aug_.col(0)  = x_aug_;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug_.col(i+1)       = x_aug_ + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_+n_aug_) * L.col(i);
  }
  std::cout << "Augumented Sigma Points: " << std::endl << Xsig_aug_ << std::endl;
}

/**
*
* Predict sigma points on every new state and covariance.
*/
void UKF::predictSigmaPoints(double delta_t) {
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;  
  }
}

void UKF::predictMeasurment(Eigen::MatrixXd R) {
  //mean predicted measurement
  z_pred_.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred_ = z_pred_ + weights_(i) * Zsig_.col(i);
  }

  //measurement covariance matrix S
  S_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  S_ = S_ + R;
}

void UKF::UpdateMeasurment(MeasurementPackage meas_package) {

  VectorXd z = meas_package.raw_measurements_;
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S_.inverse();
  //residual
  VectorXd z_diff = z - z_pred_;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S_*K.transpose();
  double nis = z_diff.transpose() * S_.inverse() * z_diff;
  if (meas_package.sensor_type_ == MeasurementPackage::LASER){
    NIS_lidar = nis;
  } else {
    NIS_radar = nis;
  }
  
}
