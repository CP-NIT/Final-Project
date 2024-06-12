import numpy as np
import unittest

# تابع محاسبه میدان الکتریکی در یک نقطه مشخص
def electric_field(charge_position, charge_value, point):
    """
    محاسبه میدان الکتریکی در یک نقطه مشخص توسط یک بار الکتریکی
    """
    k = 8.99e9  # N*m^2/C^2
    r = np.array(point) - np.array(charge_position)
    r_mag = np.linalg.norm(r)
    r_hat = r / r_mag
    E = (k * charge_value / r_mag**2) * r_hat
    return E

# تابع محاسبه میدان الکتریکی کل در یک نقطه مشخص
def total_electric_field(charges, point):
    """
    محاسبه میدان الکتریکی کل در یک نقطه مشخص توسط چند بار الکتریکی
    """
    E_total = np.array([0.0, 0.0])
    for charge in charges:
        E_total += electric_field(charge['position'], charge['value'], point)
    return E_total

# تابع محاسبه میدان الکتریکی از یک خط بار یکنواخت
def electric_field_line(charge_density, line_length, point):
    """
    محاسبه میدان الکتریکی از یک خط بار یکنواخت در یک نقطه مشخص
    """
    k = 8.99e9  # N*m^2/C^2
    E_total = np.array([0.0, 0.0])
    num_segments = 100  # تعداد بخش‌ها برای تقریب خط بار
    dl = line_length / num_segments
    for i in range(num_segments + 1):
        charge_position = np.array([i * dl, 0])
        charge_value = charge_density * dl
        E_total += electric_field(charge_position, charge_value, point)
    return E_total

# استفاده نمونه
charges = [{'position': [0, 0], 'value': 1e-9}, {'position': [1, 0], 'value': -1e-9}]
point = [0.5, 0.5]

E_total = total_electric_field(charges, point)
# ذخیره میدان الکتریکی کل به فایل
with open('total_electric_field.txt', 'w') as f:
    np.savetxt(f, [E_total])

charge_density = 1e-9
line_length = 1.0
E_line = electric_field_line(charge_density, line_length, point)
# ذخیره میدان الکتریکی از خط به فایل
with open('electric_field_line.txt', 'w') as f:
    np.savetxt(f, [E_line])

print("میدان الکتریکی کل در نقطه {}: {}".format(point, E_total))
print("میدان الکتریکی از خط در نقطه {}: {}".format(point, E_line))

class TestElectricField(unittest.TestCase):
    def test_electric_field(self):
        charge_position = [0, 0]
        charge_value = 1e-9
        point = [1, 0]

        expected_E = np.array([8.99e9 * charge_value, 0])

        E = electric_field(charge_position, charge_value, point)

        self.assertTrue(np.allclose(E, expected_E))

if __name__ == '__main__':
    unittest.main()