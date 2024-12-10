public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        Bus bus = new Bus();
        Card studentCard = new StudentCard();
        Card oldCard = new OldCard();
        new GeneralCard();
        bus.useCard(studentCard);
        bus.useCard(oldCard);
    }
}
