class Female extends Person {
    Female() {
    }

    public void Sing() {
        System.out.print("女生请唱 [默认分贝] ");
    }

    public void Sing(int volume) {
        System.out.print(String.format("女生唱歌 [%d分贝] ", volume));
    }
}
