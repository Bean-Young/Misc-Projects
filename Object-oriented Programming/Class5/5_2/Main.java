public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        Rodent[] animals = new Rodent[]{new Rodent() {
        }, new Mouse(), new Gebil(), new Hamster()};
        Rodent[] var2 = animals;
        int var3 = animals.length;

        for(int var4 = 0; var4 < var3; ++var4) {
            Rodent r = var2[var4];
            r.Eat();
        }

    }
}
