#include <iostream>
#include <string>

namespace drl {

class RosBridge {
 public:
  void Connect(const std::string& sensor_topic, const std::string& actuator_topic) {
    std::cout << "Connecting to sensor topic: " << sensor_topic << std::endl;
    std::cout << "Connecting to actuator topic: " << actuator_topic << std::endl;
  }

  void ForwardAction(const std::string& action) {
    std::cout << "Forwarding action: " << action << std::endl;
  }
};

}  // namespace drl

int main() {
  drl::RosBridge bridge;
  bridge.Connect("/sensors/state", "/actuators/cmd");
  bridge.ForwardAction("CMD:0.5");
  return 0;
}
