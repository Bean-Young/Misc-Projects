public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        int[] nums = new int[]{1, 2, 3, 4, 5};
        int index = 5;

        int num;
        try {
            num = nums[index];
        } catch (ArrayIndexOutOfBoundsException var8) {
            num = index;
        } finally {
            index = 0;
        }

        System.out.println(num);
        System.out.println(index);
    }
}
