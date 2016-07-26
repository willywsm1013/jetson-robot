const int left0 = 2;
const int left1 = 3;
const int right0 = 4;
const int right1 = 5;

void setup() {
 pinMode(left0, OUTPUT);
 pinMode(left1, OUTPUT);
 pinMode(right0, OUTPUT);
 pinMode(right1, OUTPUT);
 Serial.begin(9600);
}

void forward(int speed) {
  digitalWrite(left0, LOW);
  analogWrite(left1, speed);
  digitalWrite(right0, LOW);
  analogWrite(right1, speed);
}

void backward(int speed) {
  digitalWrite(left0, HIGH);
  analogWrite(left1, 255-speed);
  digitalWrite(right0, HIGH);
  analogWrite(right1, 255-speed);
}

void loop() {
  forward(255);
  /*
  for (int i=0; i<256; i++) {
    Serial.println(i);
    forward(i);
    delay(100);
  }
  */
}
