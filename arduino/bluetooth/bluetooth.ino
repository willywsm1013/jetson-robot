/*
 * Title:        Bluetooth Remote Control
 * Description:  This program is intented to work with a connected
 *               Bluetooth dongle to the robot and an Android phone
 *               running Blueberry application
 * Author:       Karl Kangur <karl.kangur@gmail.com>
 * Version:      1.0
 * Website:      github
 */
#define LED 2

char serialData[32];
byte com = 0, error = 0, timerCounter = 0;
boolean connected;

void setup()
{
  pinMode(LED, OUTPUT);
  // the bluetooth dongle communicates at 9600 baud only
  Serial.begin(9600);
  Serial1.begin(38400);
  Serial.println("Start");
}

void loop()
{
  if(Serial1.available() > 0)
  {
    digitalWrite(LED, HIGH);
    connected = true;
    // clear timeout
    com = timerCounter;

    Serial1.readBytesUntil('\n', serialData, 31);
    Serial.println(serialData);
    switch(serialData[0])
    {
      Serial.println(serialData);
    case 0:
      Serial1.println(0);
      break;
    case 'a':
      // use as a small and slow oscilloscope
      int pin;
      if(parseCommand(serialData, &pin, 1) && pin >= 0 && pin <= 7)
      {
        // stop loop by sending something to the robot
        while(!Serial1.available() && connected)
        {
          Serial1.println(analogRead(pin));
        }
      }
      else
      {
        Serial1.println("Error while setting ADC pin");
      }
      break;
    case 's':
      // set left and right motor speeds
      int speed[2];
      if(parseCommand(serialData, speed, 2))
      {
        setSpeed(speed[0], speed[1]);
        Serial1.println("New speed set");
      }
      else
      {
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
}

void reset()
{
  connected = false;
  setSpeed(0, 0);
}

/**
 * This function makes ints out of the received serial data, the 2 first
 * characters are not counted as they consist of the command character and
 * a comma separating the first variable.
 *
 * @params command The whole serial data received as an address
 * @params returnValues The array where the return values go as an address
 * @params returnNumber The number of values to set inside the returnValues variable
 */
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

ISR(TIMER2_OVF_vect)
{
}

void setSpeed(int left, int right)
{
}



