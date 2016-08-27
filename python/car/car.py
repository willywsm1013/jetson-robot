class Car:
    def __init__(self, arduino):
        self.arduino = arduino
        self.lastAngle = 0
        self.p = 0.5
        self.i = 0
        self.d = 1
        self.accum = 0
        self.rSpeed = 0
        self.lSpeed = 0
        self.gamma = 0.5

    def forward(self, angle):
        self.accum += angle
        self.rSpeed = 20 - self.p * angle - self.i * self.accum - self.d * (angle-self.lastAngle)
        self.lSpeed = 20 + self.p * angle + self.i * self.accum + self.d * (angle-self.lastAngle)
        self.toArduino()

    def setSpeed(self, right, left):
        self.rSpeed = right
        self.lSpeed = left
        """
        self.rSpeed += self.gamma * (right - self.rSpeed)
        self.lSpeed += self.gamma * (left - self.lSpeed)
        """
        self.toArduino()

    def rotate(self, angle):
        self.rSpeed = 10
        self.lSpeed = 10
        if angle > 0:
            self.rSpeed *= -1
        elif angle < 0:
            self.lSPeed += -1
        else:
            self.rSpeed, self.lSpeed = 20, 20
        self.toArduino()

    def stop(self):
        self.rSpeed = 0
        self.lSpeed = 0
        self.toArduino()

    def toArduino(self):
        if self.rSpeed < -90:
            self.rSpeed = -90
        elif self.rSpeed > 90:
            self.rSpeed = 90
        if self.lSpeed < -90:
            self.lSpeed = -90
        elif self.lSpeed > 90:
            self.lSpeed = 90
        command = 's ' + str(int(self.rSpeed)) + ',' + str(int(self.lSpeed)) + '\n'
        if self.arduino.available() == True:
            self.arduino.push(command)
