#include <Arduino.h>
#include <Servo.h>

#define LEFT  5
#define RIGHT  3

Servo LServo;
Servo RServo;
unsigned long timeStart=0;
unsigned long timePass;
char serialData[32];
int cnt=0;

void setup() {
    pinMode(LEFT, OUTPUT);
    pinMode(RIGHT, OUTPUT);
    LServo.attach(LEFT);
    RServo.attach(RIGHT);
    Serial.begin(9600);
}

void setSpeed(int RSpeed,int LSpeed){
    LServo.write(90-LSpeed);
    RServo.write(RSpeed+90);
}

bool parseCommand(char* command, int* returnValues, byte returnNumber)
{
    // parsing state machine
    byte i = 1, j = 0, sign = 0, ch = 0, number;
    int temp = 0;
    while(i++){
        switch(*(command + i)){
            case '\0':
            case ',':
                if(ch != 0){
                    returnValues[j++] = sign?-temp:temp;
                    sign = 0;
                    temp = 0;
                    ch = 0;
                }
                else{
                    return false;
                }
                break;
            case '-':
                sign = 1;
                break;
            default:
                // convert string to int
                number = *(command + i) - '0';
                if(number < 0 || number > 9){
                    return false;
                }
                temp = temp * 10 + number;
                ch++;
        }
        // enough return values have been set
        if(j == returnNumber){
            return true;
        }
        // end of command reached
        else if(*(command + i) == '\0'){
            return false;
        }
    }
}

void loop() {
    if (Serial.available() > 0) {
        int length = Serial.readBytesUntil('\n', serialData, 31);
        serialData[length] = '\0';
        Serial.println(serialData);
        Serial.println(cnt);
        cnt++;
        switch(serialData[0]){
            case 's' :
                int speed[2];
                timePass=millis()-timeStart;
                timeStart=millis();
                //timeDelay=timePass*0.75+timeDelay*0.15;
                if(timePass > 100) timePass=100;
                //Serial.print("time : ");
                //Serial.println(timePass);
                if(parseCommand(serialData,speed,2)){
                    setSpeed(speed[0],speed[1]);
                    delay( timePass/2 + timePass/4);
                    //setSpeed(90,90);
                }
                break;
        }
        //Serial.println("Finish");
    }
}
