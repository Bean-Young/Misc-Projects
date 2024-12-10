class Auto {
    private int tireCount;
    private String color;
    private double weight;
    private double speed;

    public Auto(int tireCount, String color, double weight, double speed) {
        this.tireCount = tireCount;
        this.color = color;
        this.weight = weight;
        this.speed = speed;
    }

    public Auto(String color, double weight) {
        this(4, color, weight, 0.0);
    }

    public void start() {
        System.out.print("");
    }

    public void stop() {
        System.out.print("");
    }

    public int getTireCount() {
        return this.tireCount;
    }

    public void setTireCount(int tireCount) {
        this.tireCount = tireCount;
    }

    public String getColor() {
        return this.color;
    }

    public void setColor(String color) {
        this.color = color;
    }

    public double getWeight() {
        return this.weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getSpeed() {
        return this.speed;
    }

    public void setSpeed(double speed) {
        this.speed = speed;
    }
}
