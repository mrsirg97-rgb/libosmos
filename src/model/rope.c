#include <math.h>

#include "osmos/model/rope.h"

/* ref.py's freqs_cis, each step a float32 operation as torch's (the corrections in double, as python's math) */
bool rope_freqs(const Rope *r, float *freqs)
{
  if (!r || !freqs || r->dim < 2 || r->dim % 2 || r->dim > ROPE_MAX_DIM || !(r->base > 1))
    return false;
  if (r->original > 0 && !(r->factor > 0 && r->beta_fast > 0 && r->beta_slow > 0))
    return false;
  const int half = r->dim / 2;
  for (int i = 0; i < half; i++)
  {
    const float e = (float)(2 * i) / (float)r->dim;
    freqs[i] = 1.0f / powf(r->base, e);
  }
  if (r->original <= 0)
    return true;
  const double two_pi = 2 * M_PI;
  const double lb = 2 * log(r->base);
  const double c_fast = r->dim * log(r->original / (r->beta_fast * two_pi)) / lb;
  const double c_slow = r->dim * log(r->original / (r->beta_slow * two_pi)) / lb;
  const int low = (int)fmax(floor(c_fast), 0);
  const int high = (int)fmin(ceil(c_slow), r->dim - 1);
  const float span = high - low > 0 ? (float)(high - low) : 1e-3f;
  for (int i = 0; i < half; i++)
  {
    float ramp = ((float)i - (float)low) / span;
    ramp = ramp < 0 ? 0 : ramp > 1 ? 1 : ramp;
    const float smooth = 1.0f - ramp;
    const float a = freqs[i] / r->factor * (1.0f - smooth);
    const float b = freqs[i] * smooth;
    freqs[i] = a + b;
  }
  return true;
}
