const int motor11 = 10;
const int motor12 = 9;
const int motor21 = 6;
const int motor22 = 5;

void setup() {
  // put your setup code here, to run once:
  pinMode(motor11, OUTPUT);
  pinMode(motor12, OUTPUT);
  pinMode(motor21, OUTPUT);
  pinMode(motor22, OUTPUT);
  Serial.begin(9600);
}

void go(int left, int right) {
  
  if (right < 0) {
    analogWrite(motor11, 0);
    analogWrite(motor12, -right);
  }
  else {
    analogWrite(motor11, right);
    analogWrite(motor12, 0);
  }
  if (left < 0) {
    analogWrite(motor21, 0);
    analogWrite(motor22, -left);
  }
  else {
    analogWrite(motor21, left);
    analogWrite(motor22, 0);
  }
}


int mode = 0;
int left_min = 1024, left_max = 0, left_thred;
int right_min = 1024, right_max = 0, right_thred;
int sensor_left, sensor_right;
int last_error = 0;

int left_speed, right_speed;

void get_sensor(bool avg = false) {
  int tmp_left = analogRead(0);
  int tmp_right = analogRead(1);
  if (avg) {
    sensor_right = (sensor_right  + tmp_right*3) / 4;
    sensor_left = (sensor_left  + tmp_left*3) / 4;
  }
  else {
    sensor_right = tmp_right;
    sensor_left = tmp_left;
  }
}

void print_sensor() {
  Serial.println(String(sensor_left) + " "+String(sensor_right));
}

void update_minmax() {
  if (sensor_right < right_min) right_min = sensor_right;
  if (sensor_right > right_max) right_max = sensor_right;
  if (sensor_left < left_min) left_min = sensor_left;
  if (sensor_left > left_max) left_max = sensor_left;
  Serial.println(String(left_min) + " " +String(left_max) + " "+String(right_min) + " "+String(right_max) + " ");
}

void adjust_sensor() {
  if (sensor_left < (left_min * 3 + left_max) / 4) left_min = (left_min * 3 + sensor_left) / 4;
  else if (sensor_left > (left_min + left_max * 3) / 4) left_max = (left_max * 3 + sensor_left) / 4;
  if (sensor_right < (right_min *3 + right_max) / 4) right_min = (right_min * 3 + sensor_right) / 4;
  else if (sensor_right > (right_min + right_max * 3) / 4) right_max = (right_max * 3 + sensor_right) / 4;
}

void loop() {
  if (mode == -1) {
    get_sensor();
    print_sensor();
    go(255, 0);
    //delay(1000);
    go(0, 255);
    //delay(1000);
    /*
    go(-255, 0);
    delay(1000);
    go(0, -255);
    delay(1000);
    */
  }
  if (mode == 0) {
    get_sensor();
    update_minmax();
    
    while(left_max - left_min < 200) {
      get_sensor();
      update_minmax();
    }
    while(right_max - right_min < 200) {
      get_sensor();
      update_minmax();
    }
    left_speed = 255, right_speed = 255;
    go(left_speed, right_speed);
    mode = 1;
  }
  else if (mode == 1) {
    get_sensor(true);
    bool left = sensor_left < (left_min + left_max) / 2;
    bool right = sensor_right < (right_min + right_max) / 2;
    Serial.println(String(left) + String(right));

    if (left) go(-80, 130);
    else if (right) go(130, -80);
    else go(130, 130);
  }
}
