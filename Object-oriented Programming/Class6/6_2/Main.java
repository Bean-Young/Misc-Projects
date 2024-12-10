public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        Auto auto1 = new Auto(4, "Red", 1500.0, 120.0);
        Auto auto2 = new Auto("Blue", 1300.0);
        Car car = new Car(4, "Black", 1600.0, 150.0, true, true);
        auto1.start();
        auto1.stop();
        auto2.start();
        auto2.stop();
        car.start();
        car.stop();
    }
}
