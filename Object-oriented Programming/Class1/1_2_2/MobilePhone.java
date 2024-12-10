public class MobilePhone extends Phone {
    private String brand;
    private String ownerId;

    public MobilePhone(String brand, String code, String ownerId) {
        super(code);
        this.brand = brand;
        this.ownerId = ownerId;
    }

    public void display() {
        super.display();
        System.out.println("Brand=" + this.brand);
        System.out.println("OwnerId=" + this.ownerId);
    }
}
