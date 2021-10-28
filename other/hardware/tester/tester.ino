#include <DirectIO.h>
Output<2> st1;
Output<3> dir1;
Output<4> st2;
Output<5> dir2;
Output<6> st3;
Output<7> dir3;
Output<8> st4;
Output<9> dir4;

Output<13> stepBlock;

const byte numChars = 32;
char receivedChars[numChars];
char tempChars[numChars];
int integerFromPC1 = 0;
int integerFromPC2 = 0;
int integerFromPC3 = 0;
int integerFromPC4 = 0;
int exeTime = 0;
int accelTime;
boolean newData = false;
const byte degStep = 18; // 1.8 deg per step
const byte microstep = 8; // 1,2,4,8,16
unsigned int startU; // acceleration start velocity (ms)

int currentStep1 = 0;// memory of current location
int currentStep2 = 0;
int currentStep3 = 0;
int currentStep4 = 0;

void setup() {
    stepBlock = HIGH;
    Serial.begin(9600);
    Serial.println("<300, 300, 300, 300, 700>");
    //Serial.println("<1, 2, 3, 4, msec>");
    //Serial.println();
}


void loop() {
    recvWithStartEndMarkers();
    if (newData == true) {
        strcpy(tempChars, receivedChars);
        parseData();
        newData = false;
        if (integerFromPC1==7777) {
          //Serial.println("1case");
          stepBlock = LOW;
          delay(200);
          setHigh();
          bom(1000, deg2step(50), deg2step(20), deg2step(100), 0, 2000, 2000, 2000, 2000);
        } else if (integerFromPC1==6666) {
          //Serial.println("2case");
          stepBlock = HIGH;
        } else {
          //Serial.println("3case");
          exe(integerFromPC1,integerFromPC2,integerFromPC3,integerFromPC4,(float)exeTime);
        }
        
    }
}


void recvWithStartEndMarkers() {
    static boolean recvInProgress = false;
    static byte ndx = 0;
    char startMarker = '<';
    char endMarker = '>';
    char rc;

    while (Serial.available() > 0 && newData == false) {
        rc = Serial.read();

        if (recvInProgress == true) {
            if (rc != endMarker) {
                receivedChars[ndx] = rc;
                ndx++;
                if (ndx >= numChars) {
                    ndx = numChars - 1;
                }
            }
            else {
                receivedChars[ndx] = '\0'; // terminate the string
                recvInProgress = false;
                ndx = 0;
                newData = true;
            }
        }

        else if (rc == startMarker) {
            recvInProgress = true;
        }
    }
}


void parseData() {      // split the data into its parts

    char * strtokIndx; // this is used by strtok() as an index

    strtokIndx = strtok(tempChars,",");
    integerFromPC1 = atoi(strtokIndx);
 
    strtokIndx = strtok(NULL, ",");
    integerFromPC2 = atoi(strtokIndx);

    strtokIndx = strtok(NULL, ",");
    integerFromPC3 = atoi(strtokIndx);

    strtokIndx = strtok(NULL, ",");
    integerFromPC4 = atoi(strtokIndx);

    strtokIndx = strtok(NULL, ",");
    exeTime = atol(strtokIndx);

}

// transform degrees to number of steps, using microstep int
int deg2step(int deg) {
        return round( (float)deg / degStep * microstep );
    }

void bom(int exeTime, int cnt1, int cnt2, int cnt3, int cnt4, long u1, long u2, long u3, long u4) {
  //Serial.println("bom fun");
          unsigned int cn1 = 0;
          unsigned int cn2 = 0;
          unsigned int cn3 = 0;
          unsigned int cn4 = 0;
          unsigned long lastExe1Time = micros();
          unsigned long lastExe2Time = micros();
          unsigned long lastExe3Time = micros();
          unsigned long lastExe4Time = micros();
          unsigned long currentDelay1 = (long)u1;
          unsigned long currentDelay2 = (long)u2;
          unsigned long currentDelay3 = (long)u3;
          unsigned long currentDelay4 = (long)u4;
          unsigned long startTime = millis();
          //Serial.println(cnt3);Serial.println(u3);
          
          while (cn1<cnt1 || cn2<cnt2 || cn3<cnt3 || cn4<cnt4) {
            //Serial.println(cn1);Serial.println(cn2);
            //Serial.println(cn3);//Serial.println(cn4);
            if ((micros() - lastExe1Time > currentDelay1) && (cn1<cnt1)) {
              lastExe1Time = micros();
              cn1++;
              
              st1 = HIGH;
              delayMicroseconds(5);
              st1 = LOW;
            }
            
            if ((micros() - lastExe2Time > currentDelay2) && (cn2<cnt2)) {
              lastExe2Time = micros();
              cn2++;
              
              st2 = HIGH;
              delayMicroseconds(5);
              st2 = LOW;
            }
            
            if ((micros() - lastExe3Time > currentDelay3) && (cn3<cnt3)) {
              lastExe3Time = micros();
              cn3++;
              
              st3 = HIGH;
              delayMicroseconds(5);
              st3 = LOW;
            }

            if ((micros() - lastExe4Time > currentDelay4) && (cn4<cnt4)) {
              lastExe4Time = micros();
              cn4++;
              
              st4 = HIGH;
              delayMicroseconds(5);
              st4 = LOW;
            }
          }
          //Serial.println("done");
          
          
        }
void setHigh() {
    dir1 = HIGH;
    dir2 = HIGH;
    dir3 = HIGH;
    dir4 = HIGH;
}

void setLow() {
    dir1 = LOW;
    dir2 = LOW;
    dir3 = LOW;
    dir4 = LOW;
}

void exe(int deg1, int deg2, int deg3, int deg4, float exeTime) {
    long time1 = 0;
    long time2 = 0;
    long time3 = 0;
    long time4 = 0;
    currentStep1!=0 ? time1 = abs((long)exeTime*1000/currentStep1) :0;
    currentStep2!=0 ? time2 = abs((long)exeTime*1000/currentStep2) :0;
    currentStep3!=0 ? time3 = abs((long)exeTime*1000/currentStep3) :0;
    currentStep4!=0 ? time4 = abs((long)exeTime*1000/currentStep4) :0;
    stepBlock = LOW;
    setLow();
    Serial.println("goin down");
    Serial.println(currentStep1);
    Serial.println(currentStep2);
    Serial.println(currentStep3);
    Serial.println(currentStep4);
//    Serial.println(time1);
//    Serial.println(time2);
//    Serial.println(time3);
//    Serial.println(time4);
    bom(exeTime, currentStep1, currentStep2, currentStep3, currentStep4, time1, time2, time3, time4); // go to 0's
    
    int delta1 = deg2step(deg1);
    int delta2 = deg2step(deg2);
    int delta3 = deg2step(deg3);
    int delta4 = deg2step(deg4);
    currentStep1 = delta1;
    currentStep2 = delta2;
    currentStep3 = delta3;
    currentStep4 = delta4;
    delta1!=0 ? time1 = abs((long)exeTime*1000/currentStep1) :0;
    delta2!=0 ? time2 = abs((long)exeTime*1000/currentStep2) :0;
    delta3!=0 ? time3 = abs((long)exeTime*1000/currentStep3) :0;
    delta4!=0 ? time4 = abs((long)exeTime*1000/currentStep4) :0;
    //st1 = LOW;
    //st2 = LOW;
    setHigh();
    Serial.println("goin up");
    Serial.println(delta1);
    Serial.println(delta2);
    Serial.println(delta3);
    Serial.println(delta4);
//    Serial.println(time1);
//    Serial.println(time2);
//    Serial.println(time3);
//    Serial.println(time4);
    Serial.println("");
    bom(exeTime, delta1, delta2, delta3, delta4, time1, time2, time3, time4);
}
