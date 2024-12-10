class Car extends Auto {
    private boolean hasAudio;
    private boolean hasCDPlayer;

    public Car(int tireCount, String color, double weight, double speed, boolean hasAudio, boolean hasCDPlayer) {
        super(tireCount, color, weight, speed);
        this.hasAudio = hasAudio;
        this.hasCDPlayer = hasCDPlayer;
    }

    public void start() {
    }

    public void stop() {
    }

    public boolean isHasAudio() {
        return this.hasAudio;
    }

    public void setHasAudio(boolean hasAudio) {
        this.hasAudio = hasAudio;
    }

    public boolean isHasCDPlayer() {
        return this.hasCDPlayer;
    }

    public void setHasCDPlayer(boolean hasCDPlayer) {
        this.hasCDPlayer = hasCDPlayer;
    }
}
