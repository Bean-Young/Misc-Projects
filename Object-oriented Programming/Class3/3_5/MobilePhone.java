class MobilePhone extends Phone {
    private int time;
    private double price;

    public MobilePhone(String code, int time, double price) {
        super(code);
        this.time = time;
        this.price = price;
    }

    public double pay() {
        return (double)this.time * this.price;
    }
}
