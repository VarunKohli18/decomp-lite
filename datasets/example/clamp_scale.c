static double clamp(double v, double lo, double hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

double clamp_scale(double value, double scale)
{
    return clamp(value * scale, -1.0, 1.0);
}
