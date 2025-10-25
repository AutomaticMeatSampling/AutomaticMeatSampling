char receivedChar = ' ';
int sensorPin = A0; 
int voltageOut = A1;
int sensorValue = 0;

#if defined(ARDUINO_SAMD_ZERO) && defined(SERIAL_PORT_USBVIRTUAL)
  // Required for Serial on Zero based boards
  #define Serial SERIAL_PORT_USBVIRTUAL
#endif

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinmode(voltageOut, OUTPUT)
  pinmode(sensorPin, INPUT)
}

void loop() {
  // put your main code here, to run repeatedly:
  boolean connect;
  int procNumber = 0;
  procNumber = verifyCOM();

  if (procNumber == 2){
    disconnectCheck();
  }

}

void disconnectCheck(){
  int timeCheck = 0; // needs to be disconnected for 300ms
  boolean connect = true;
  digitalWrite(voltageOut, HIGH);
  while (timeCheck < 3){
    sensorValue = analogRead(sensorPin);  // read analog voltage
    if (sensorValue == 0){
      timeCheck++;
      delay(100);
    }
  }
  //digitalWrite(voltageOut, LOW);
  Serial.println("DISCONNECTED");  // let python code know that the electrical contacts have disconnected

}

int verifyCOM(){
  int procNum = 0;  // 1 means doing handshake verification with python, 2 means checking for rail contact being disconnected
  
  if (Serial.available() > 0){
    receivedChar = Serial.read();
    if (receivedChar == 'R'){ // R for rail system microcontroller
      Serial.print("R here \n");
      procNum = 1;
    }
    else if (receivedChar == 'D'){  // D to check if disconnected
      procNum = 2;
    }
  }
  return procNum;
}
