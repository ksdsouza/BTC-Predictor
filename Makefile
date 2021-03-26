
clean:
	mvn clean

package:
	$(MAKE) clean
	mvn package

