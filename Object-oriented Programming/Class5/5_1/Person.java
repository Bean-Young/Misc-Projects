abstract class Person {
    Person() {
    }

    public void Sing() {
        System.out.print("请唱 [默认分贝] ");
    }

    public void Sing(int volume) {
        System.out.print(String.format("唱歌 [%d分贝] ", volume));
    }
}
