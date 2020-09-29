#!/usr/bin/env python
# coding: utf-8

import imgaug.augmenters as iaa


def data_aug(images):
    seq = iaa.Sometimes(
        0.5,
        iaa.Identity(),
        iaa.Sometimes(0.5,
                      iaa.Sequential([
                          iaa.Fliplr(0.5),
                          iaa.Sometimes(0.5,
                                        iaa.OneOf([
                                            iaa.Add((-40, 40)),
                                            iaa.AddElementwise((-40, 40)),
                                            iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)),
                                            iaa.AdditiveLaplaceNoise(scale=(0, 0.2*255)),
                                            iaa.AdditivePoissonNoise((0, 40)),
                                            iaa.MultiplyElementwise((0.5, 1.5)),
                                            iaa.ReplaceElementwise(0.1, [0, 255]),
                                            iaa.SaltAndPepper(0.1)
                                        ])
                                       ),
                          iaa.OneOf([
                              iaa.Cutout(nb_iterations=2,size=0.15,cval=0,squared=False),
                              iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
                              iaa.Dropout(p=(0, 0.2)),
                              iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1)),
                              iaa.Cartoon(),
                              iaa.BlendAlphaVerticalLinearGradient(iaa.TotalDropout(1.0),min_value=0.2, max_value=0.8),
                              iaa.GaussianBlur(sigma=(0.0, 3.0)),
                              iaa.AverageBlur(k=(2, 11)),
                              iaa.MedianBlur(k=(3, 11)),
                              iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
                              iaa.MotionBlur(k=20),
                              iaa.AllChannelsCLAHE(),
                              iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
                              iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                              iaa.Affine(scale=(0.5, 1.5)),
                              iaa.Affine(translate_px={"x": (-20, 20), "y": (-20, 20)}),
                              iaa.Affine(shear=(-16, 16)),
                              iaa.pillike.EnhanceSharpness()
                          ]),
                          iaa.OneOf([
                              iaa.GammaContrast((0.5, 2.0)),
                              iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
                              iaa.LogContrast(gain=(0.6, 1.4)),
                              iaa.LinearContrast((0.4, 1.6)),
                              iaa.pillike.EnhanceBrightness()
                          ])
                      ]),
                      iaa.Sometimes(0.5,
                                    iaa.RandAugment(n=2, m=9),
                                    iaa.RandAugment(n=(0, 3),m=(0,9))
                                   )
                     )
    )
    images = seq(images = images)
    return images