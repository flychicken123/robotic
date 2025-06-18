public class PackageSorter {
    private static final int MAX_VOLUME = 1_000_000;
    private static final int MAX_DIMENSION = 150;
    private static final int MAX_MASS = 20; 

    public static String sort(int width, int height, int length, int mass) {
        boolean isBulky = isBulky(width, height, length);
        boolean isHeavy = isHeavy(mass);

        if (isBulky && isHeavy) {
            return "REJECTED";
        } else if (isBulky || isHeavy) {
            return "SPECIAL";
        } else {
            return "STANDARD";
        }
    }

    private static boolean isBulky(int width, int height, int length) {
        if (width >= MAX_DIMENSION || 
            height >= MAX_DIMENSION || 
            length >= MAX_DIMENSION) {
            return true;
        }

        long volume = (long) width * height * length;
            System.out.println("volume: " + volume);
            return volume >= MAX_VOLUME;
    }

    private static boolean isHeavy(int mass) {
        return mass >= MAX_MASS;
    }

    public static void main(String[] args) {
        System.out.println(sort(90, 100, 100, 10));  // STANDARD
        System.out.println(sort(150, 100, 100, 10));  // SPECIAL (bulky)
         System.out.println(sort(140, 140, 140, 10));  // SPECIAL (bulky)
        System.out.println(sort(100, 100, 100, 20));  // SPECIAL (heavy)
        System.out.println(sort(150, 100, 100, 20));  // REJECTED (both bulky and heavy)
    }
} 