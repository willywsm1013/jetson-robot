#define ECHOPIN 12        // Orange line
#define TRIGPIN 13        // Red line
#define SERVOPIN 9
#define DETECT_POS_S 10
#define DETECT_POS_E 170
#define POS_INC 20
#define WAIT_S 80

#include <Servo.h>
Servo myservo; // create servo object to control a servo

int pos = 0; // variable to store the servo position
float dis;
float pre[(DETECT_POS_E-DETECT_POS_S)/POS_INC+1];
float now[(DETECT_POS_E-DETECT_POS_S)/POS_INC+1];

void setup() 
{
  Serial.begin(9600); 
  myservo.attach(SERVOPIN); // attaches the servo on pin 9 to the servo object
  pinMode(ECHOPIN, INPUT); 
  pinMode(TRIGPIN, OUTPUT); 
  myservo.write(DETECT_POS_S); // tell servo to go to position in variable 'pos'
  delay(100);
  for (int i=0; i<=(DETECT_POS_E-DETECT_POS_S)/POS_INC; i++)
    pre[i] = 100;
}

void Print (float dis , int deg)
{
   // R for distance in cm, T for pos in degree
        Serial.print(dis);Serial.print(", ");
         Serial.print(deg);Serial.println(".");
   //delay(100);
}

float Distance () 
{  
  digitalWrite(TRIGPIN, LOW); 
  delayMicroseconds(2); 
  digitalWrite(TRIGPIN, HIGH); 
  delayMicroseconds(100); 
  digitalWrite(TRIGPIN, LOW);
  
  // Distance Calculation
  float Time=pulseIn(ECHOPIN, HIGH,6000);
  float distance=100;
  if(Time)
    distance = (Time / 29.4) / 2; // return in microseconds, 1cm requires 29.4ms
  return (distance);
}

float pre_x, pre_y, now_x, now_y;

void loop() 
{
  now_x = now_y = 0;
  for (pos = DETECT_POS_S; pos <= DETECT_POS_E; pos += POS_INC) 
  { 
    myservo.write(pos); // tell servo to go to position in variable 'pos'
    delay(WAIT_S); // waits some time for the servo to reach the position
    dis=(Distance()+Distance()+Distance()+Distance()+Distance()+Distance())/6;
    //Serial.print(dis);
    //Serial.print(" ");
    now_x += dis * cos(pos * PI / 180);
    now_y += dis * sin(pos * PI / 180);
  }
  //Serial.println();
  
  for (pos = DETECT_POS_E; pos>=DETECT_POS_S; pos=pos-POS_INC) 
  {
    myservo.write(pos); // tell servo to go to position in variable 'pos'
    delay(WAIT_S); // waits some time for the servo to reach the position
    dis=(Distance()+Distance()+Distance()+Distance()+Distance()+Distance())/6;
    //Serial.print(dis);
    //Serial.print(" ");  
    now_x += dis * cos(pos * PI / 180);
    now_y += dis * sin(pos * PI / 180);
  }
  
  //Serial.println();
  
  Serial.print((now_x - pre_x)/16);
  Serial.print(" ");
  Serial.println((now_y - pre_y) /16);
  pre_x = now_x, pre_y = now_y;
}
