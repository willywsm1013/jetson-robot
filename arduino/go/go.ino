#include <AccelStepper.h>
#include <Servo.h>

AccelStepper motorL(AccelStepper::FULL4WIRE, 22, 23, 24, 25); 
AccelStepper motorR(AccelStepper::FULL4WIRE, 26, 27, 28, 29);

Servo servo[6];

void setup()
{
  pinMode(2, OUTPUT);
  pinMode(3, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(6, OUTPUT);
  pinMode(22, OUTPUT);
  pinMode(23, OUTPUT);
  pinMode(24, OUTPUT);
  pinMode(25, OUTPUT);
  pinMode(26, OUTPUT);
  pinMode(27, OUTPUT);
  pinMode(28, OUTPUT);
  pinMode(29, OUTPUT);

  for (int i=0; i<6; i++) servo[i].attach(i+2);
  int tmp[6] = {1462,989,940,1111,1450,1450};
  setDegree(tmp);
  
  motorL.setMaxSpeed(500);
  motorL.setAcceleration(20);
  motorR.setMaxSpeed(100);
  motorR.setAcceleration(20);

  
  Serial.begin(9600);
  Serial1.begin(38400);
  Serial.println("Start");
}

int cmd_len = 0;
char serialData[32];

void loop()
{
  bool available = false;
  if (Serial.available() > 0) {
    Serial.readBytesUntil('\n', serialData, 31);
    available = true;
  }
  else if (Serial1.available() > 0)
  {
    Serial1.readBytesUntil('\n', serialData, 31);
    available = true;
  }
  if (available) {
    Serial.println(serialData);
    switch(serialData[0]) {
    case 0:
      Serial1.println(0);
      break;
    case 'a':
      // use as a small and slow oscilloscope
      int pin;
      if(parseCommand(serialData, &pin, 1) && pin >= 0 && pin <= 7)
      {
        // stop loop by sending something to the robot
        while(!Serial1.available())
        {
          Serial1.println(analogRead(pin));
        }
      }
      else
      {
        Serial1.println("Error while setting ADC pin");
      }
      break;
    case 'w':
      // set left and right motor speeds
      int speed[2];
      if(parseCommand(serialData, speed, 2)) {
        setSpeed(speed[0], speed[1]);
        Serial1.println("New speed set");
      }
      else {
        Serial1.println("Error while setting new speed");
      }
      break;
    case 'h':
      int us[6];
      if (parseCommand(serialData, us, 6)) {
        setDegree(us);
        Serial1.println("New degree set");
      }
      else {
        Serial1.println("Error while setting new speed");
      }
      break;
    case 'i':
      // inform about robot
      Serial1.println("Zygote 1.0");
      break;
    case 'r':
      // quickly stop
      reset();
      Serial1.println("Robot reset");
      break;
    default:
      // inform user of non existing command
      Serial1.println("Command not recognised");
    }

    // clear serialData array
    memset(serialData, 0, sizeof(serialData));
  }
  motorL.runSpeed();
  motorR.runSpeed();
  //motorL.run();
}


boolean parseCommand(char* command, int* returnValues, byte returnNumber)
{
  // parsing state machine
  byte i = 1, j = 0, sign = 0, ch = 0, number;
  int temp = 0;
  while(i++)
  {
    switch(*(command + i))
    {
    case '\0':
    case ',':
      // set return value
      if(ch != 0)
      {
        returnValues[j++] = sign?-temp:temp;
        sign = 0;
        temp = 0;
        ch = 0;
      }
      else
      {
        return false;
      }
      break;
    case '-':
      sign = 1;
      break;
    default:
      // convert string to int
      number = *(command + i) - '0';
      if(number < 0 || number > 9)
      {
        return false;
      }
      temp = temp * 10 + number;
      ch++;
    }

    // enough return values have been set
    if(j == returnNumber)
    {
      return true;
    }
    // end of command reached
    else if(*(command + i) == '\0')
    {
      return false;
    }
  }
}

void reset() {
  setSpeed(0, 0);
}

void setSpeed(int left, int right) {
  motorL.setSpeed(left);
  motorR.setSpeed(right);
}

void setDegree(int us[]) {
  for (int i=0; i<6; i++) {
    servo[i].writeMicroseconds(us[i]);
  }
}

